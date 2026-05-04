import gc
import os
import queue
import threading
import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from . import Meta
from . import timing

_MEM_PROFILE = bool(os.environ.get("LORRAX_MEM_PROFILE", ""))

def _mem_report(label):
    """Print per-device memory if LORRAX_MEM_PROFILE is set. Zero cost otherwise."""
    if not _MEM_PROFILE:
        return
    gc.collect()
    s = jax.local_devices()[0].memory_stats()
    u = s.get('bytes_in_use', 0) / 1e9
    p = s.get('peak_bytes_in_use', 0) / 1e9
    if jax.process_index() == 0:
        print(f'  [MEM {label}] used={u:.3f} peak={p:.3f} GB', flush=True)
from common import jax_profile
from .cholesky_2d import (
    cholesky_2d_batched,
    dense_to_tiles,
    tiles_to_dense,
)
from .fft_helpers import (
    make_flat_k_ifftn,
    make_flat_k_fftn,
    compute_block_size_for_2d_cholesky,
)
from .load_wfns import (
    read_Gvecs_to_devices,
    load_centroids_band_chunked,
)


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


_accum_pair_density_cache = {}


def accumulate_pair_density_spin_traced(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""``P_accum + Σ_{n,s} ψ*_{n,s}(r_μ) ψ_{n,s}(r_col)`` in one JIT.

	Used by the band-chunk streaming pair-density path: each band
	chunk contributes its partial einsum to the running accumulator
	without materialising an intermediate — XLA can fuse the add
	into the einsum's output write.

	Shapes / shardings match :func:`compute_pair_density_spin_traced`
	on the two inputs; ``P_accum`` lives on the same sharding as the
	output (``P(None, 'x', 'y')``).
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('spin_traced_accum', id(mesh_xy), nk, n_rmu, nb, ns, n_col)
	if cache_key not in _accum_pair_density_cache:
		P_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		L_sharding = NamedSharding(mesh_xy, P(None, 'x', None, None))
		R_sharding = NamedSharding(mesh_xy, P(None, None, None, 'y'))

		@partial(jax.jit,
				 in_shardings=(P_sharding, L_sharding, R_sharding),
				 out_shardings=P_sharding,
				 donate_argnums=(0,))
		def _accum(P_in: jax.Array, psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return P_in + jnp.einsum('kmns,knsv->kmv', psi_L, psi_R, optimize=True)

		_accum_pair_density_cache[cache_key] = _accum
	return _accum_pair_density_cache[cache_key](P_accum, psi_rmuT_X, psi_rcol_Y)


# ============================================================================
# Vertex-parameterised pair density (bispinor / Lorentz channel generalisation)
# ============================================================================
# Generalises the spin-traced pair density to carry an arbitrary (ns, ns)
# vertex matrix V between the conjugated and non-conjugated wavefunctions:
#
#   P^V_k(μ, ν) = Σ_{n, αβ} ψ*_{n,k,α}(r_μ) V_{αβ} ψ_{n,k,β}(r_ν)
#
# With V = 1_{ns} this reduces exactly to ``compute_pair_density_spin_traced``
# (the existing implicit identity vertex is recovered).  For bispinor
# wavefunctions (ns = 4) and V = γ̃^{μ_L} ≡ γ^0 γ^{μ_L} (the matrices stored
# in ``common.gamma_matrices`` under that already-absorbed convention) this
# is the "Lorentz-channel" pair density ψ̄ γ^{μ_L} ψ used as the four ISDF
# inputs in the bispinor (DHFB) GW path.  See ``docs/BISPINOR_DHFB_DESIGN.md``
# §4.
#
# The vertex enters as a runtime jax.Array argument — one compiled function
# handles every (ns, ns) vertex of a given size, so ``mu_lorentz`` callers
# don't pay extra compile time per channel.

_compute_pair_density_vertex_cache = {}


def compute_pair_density_with_vertex(
	psi_rmuT_X: jax.Array,
	psi_rmu_Y: jax.Array,
	vertex: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Compute vertex-weighted pair density.

	    P^V_k(μ, ν) = Σ_{n, αβ} ψ*_{n,k,α}(r_μ) V_{αβ} ψ_{n,k,β}(r_ν)

	Args:
	    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None);
	        already conjugated.
	    psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y').
	    vertex:     (ns, ns) complex; replicated.  Pass jnp.eye(ns) to
	        recover the identity-vertex / spin-traced result.
	    mesh_xy:   2-D device mesh.

	Returns:
	    P_k: (nk, n_rmu, n_rmu) with P(None, 'x', 'y').
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	if vertex.shape != (ns, ns):
		raise ValueError(
			f"vertex shape {vertex.shape} does not match spinor axis ns={ns}"
		)
	cache_key = ('vertex', id(mesh_xy), nk, n_rmu, nb, ns)

	if cache_key not in _compute_pair_density_vertex_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		v_rep = NamedSharding(mesh_xy, P())
		xy_out = NamedSharding(mesh_xy, P(None, 'x', 'y'))

		@partial(jax.jit, in_shardings=(x1_4, y3_4, v_rep), out_shardings=xy_out)
		def _compute_P_vertex(psi_L, psi_R, V):
			# (k, μ, n, α) · (α, β) · (k, n, β, ν) → (k, μ, ν)
			return jnp.einsum('kmna,ab,knbv->kmv', psi_L, V, psi_R, optimize=True)

		_compute_pair_density_vertex_cache[cache_key] = _compute_P_vertex

	return _compute_pair_density_vertex_cache[cache_key](psi_rmuT_X, psi_rmu_Y, vertex)


_accum_pair_density_vertex_cache = {}


def accumulate_pair_density_with_vertex(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	vertex: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""``P_accum += Σ_{n, αβ} ψ*_{n,α}(r_μ) V_{αβ} ψ_{n,β}(r_col)`` in one JIT.

	Vertex-weighted analogue of ``accumulate_pair_density_spin_traced``.
	Layout / sharding rules match ``compute_pair_density_with_vertex``;
	``P_accum`` is donated.
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	if vertex.shape != (ns, ns):
		raise ValueError(
			f"vertex shape {vertex.shape} does not match spinor axis ns={ns}"
		)
	cache_key = ('vertex_accum', id(mesh_xy), nk, n_rmu, nb, ns, n_col)
	if cache_key not in _accum_pair_density_vertex_cache:
		P_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		L_sharding = NamedSharding(mesh_xy, P(None, 'x', None, None))
		R_sharding = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		v_rep = NamedSharding(mesh_xy, P())

		@partial(jax.jit,
				 in_shardings=(P_sharding, L_sharding, R_sharding, v_rep),
				 out_shardings=P_sharding,
				 donate_argnums=(0,))
		def _accum(P_in, psi_L, psi_R, V):
			return P_in + jnp.einsum('kmna,ab,knbv->kmv', psi_L, V, psi_R, optimize=True)

		_accum_pair_density_vertex_cache[cache_key] = _accum
	return _accum_pair_density_vertex_cache[cache_key](P_accum, psi_rmuT_X, psi_rcol_Y, vertex)


def _gamma_tilde_matrix(mu_lorentz: int) -> jax.Array:
	"""Return γ̃^{μ_L} ≡ γ^0 γ^{μ_L} as a (4, 4) complex jnp array.

	Uses the ``common.gamma_matrices`` storage convention (the matrices in
	that module are already γ^0 γ^μ — see the convention note at top of
	that file and §2.2 of ``docs/BISPINOR_DHFB_DESIGN.md``).
	"""
	from .gamma_matrices import gamma0, gamma1, gamma2, gamma3
	if mu_lorentz == 0:
		return gamma0
	if mu_lorentz == 1:
		return gamma1
	if mu_lorentz == 2:
		return gamma2
	if mu_lorentz == 3:
		return gamma3
	raise ValueError(f"mu_lorentz must be in 0..3, got {mu_lorentz}")


def compute_pair_density_lorentz(
	psi_rmuT_X: jax.Array,
	psi_rmu_Y: jax.Array,
	mu_lorentz: int,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Lorentz-channel pair density for bispinor (ns=4) wavefunctions.

	    P^{μ_L}_k(μ_c, ν_c) = Σ_{n, αβ} Ψ*_{n,k,α}(r_{μ_c}) (γ̃^{μ_L})_{αβ}
	                         Ψ_{n,k,β}(r_{ν_c})

	Phase-1 of the bispinor extension fits four ζ bases, one per μ_L ∈ {0,1,2,3},
	on the same centroid set; this helper builds the corresponding pair-density
	inputs.  See ``docs/BISPINOR_DHFB_DESIGN.md`` §4.

	Args:
	    psi_rmuT_X, psi_rmu_Y: as in ``compute_pair_density_with_vertex``
	        but with ``ns == 4`` (bispinor).
	    mu_lorentz: 0..3 — Lorentz vertex index.
	    mesh_xy: 2-D device mesh.

	Returns:
	    P^{μ_L}_k: (nk, n_rmu, n_rmu) complex on P(None, 'x', 'y').
	"""
	ns = psi_rmuT_X.shape[3]
	if ns != 4:
		raise ValueError(
			f"compute_pair_density_lorentz requires bispinor wavefunctions "
			f"(ns=4); got ns={ns}.  For 2-component spinors the existing "
			f"compute_pair_density_spin_traced is the right helper."
		)
	vertex = _gamma_tilde_matrix(mu_lorentz)
	return compute_pair_density_with_vertex(psi_rmuT_X, psi_rmu_Y, vertex, mesh_xy)


def accumulate_pair_density_lorentz(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mu_lorentz: int,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Band-chunk accumulator for ``compute_pair_density_lorentz``."""
	ns = psi_rmuT_X.shape[3]
	if ns != 4:
		raise ValueError(
			f"accumulate_pair_density_lorentz requires bispinor wavefunctions "
			f"(ns=4); got ns={ns}."
		)
	vertex = _gamma_tilde_matrix(mu_lorentz)
	return accumulate_pair_density_with_vertex(
		P_accum, psi_rmuT_X, psi_rcol_Y, vertex, mesh_xy
	)


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
		kgrid: (nkx, nky, nkz) — the 3-D form only appears inside the FFT helper.
		mesh_xy: Device mesh

	Returns:
		C_q: (nq, n_rmu, n_rmu) flat-q, P(None, 'x', 'y').
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, _ = P_l_k.shape

	cache_key = ('CCT_LR', id(mesh_xy), nk, n_rmu, nkx)

	if cache_key not in _isdf_pipeline_cache:
		flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		spec = P(None, None, None, 'x', 'y')
		local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, spec, norm='forward')
		local_fftn  = make_flat_k_fftn( mesh_xy, kgrid, spec, norm='forward')

		# Flat-k in and flat-q out (same (nk, μ, μ) shape and sharding on both
		# sides).  The 3-D k-grid reshape now lives inside ``local_{i,}fftn``,
		# so donation is safe: P_l and P_r are consumed by the two IFFTs and
		# the result preserves rank-3 end-to-end.
		@partial(jax.jit, in_shardings=(flat_xy, flat_xy), out_shardings=flat_xy,
		         donate_argnums=(0, 1))
		def _compute_CCT_LR(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			# norm='forward' for BOTH IFFT and FFT — convolution theorem with
			# unscaled IFFT (sum) + FFT/N matches gw_jax's direct k-sum:
			#   C_q = FFT(conj(IFFT(A)) ⊙ IFFT(B)) = Σ_k A*_k B_{k+q}.
			P_l_Rt = local_ifftn(P_l)
			P_r_Rt = local_ifftn(P_r)
			C_Rt = jnp.conj(P_l_Rt) * P_r_Rt
			return local_fftn(C_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR

	return _isdf_pipeline_cache[cache_key](P_l_k, P_r_k)


def compute_gram_q0_from_left_right(
	P_v_k: jax.Array,
	P_c_k: jax.Array,
	k_weights: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Build the q=0 valence-conduction pair-product Gram matrix from two
	per-k pair densities produced by ``compute_pair_density_spin_traced``.

	Mathematically (q=0 special case of the CCT-over-k structure):

	    G_{ab} = Σ_k w_k · [Σ_v φ_{v,k}(r_a)  φ*_{v,k}(r_b)]
	                     · [Σ_c ψ*_{c,k}(r_a) ψ_{c,k}(r_b)]
	           = Σ_k w_k · conj(P_v_k(a,b)) · P_c_k(a,b)

	where both ``P_v_k`` and ``P_c_k`` follow the gw_jax convention

	    P_k(μ, ν) = Σ_{n,s} ψ*_{n,k,s}(μ) · ψ_{n,k,s}(ν)

	(the ``compute_pair_density_spin_traced`` output). The ``conj`` on
	``P_v_k`` flips its conjugation pattern to the valence-projector form
	φ(a)φ*(b); multiplying it elementwise by the conduction
	ψ*(a)ψ(b) yields the valence-conduction pair-product Gram used for
	pivoted-Cholesky candidate pruning. See
	``sandbox/pivoted_cholesky.md`` §1 for the full derivation.

	Compared to ``compute_CCT_from_left_right``, this drops the k→q FFT
	pair: at q=0 the k-sum IS the answer, no convolution is needed. For
	any q≠0 you want the CCT path, not this one.

	Args:
		P_v_k: (nk, n_rmu, n_rmu) complex, valence pair density,
			P(None, 'x', 'y'). The gw_jax-convention pair density — pass
			the output of ``compute_pair_density_spin_traced`` fed with
			the valence window.
		P_c_k: (nk, n_rmu, n_rmu) complex, conduction pair density, same
			layout. Same routine, conduction window.
		k_weights: (nk,) real, k-point weights (IBZ weights summing to 1,
			or 1/nk_tot for each full-BZ k-point — whatever convention
			was used when building P_v_k / P_c_k).
		mesh_xy: ('x','y') device mesh, same one used for the pair
			densities.

	Returns:
		G: (n_rmu, n_rmu) complex Hermitian PSD, sharded P('x','y') on
			the mesh.
	"""
	nk, n_rmu, _ = P_v_k.shape
	cache_key = ('gram_q0_LR', id(mesh_xy), nk, n_rmu)

	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P('x', 'y'))
		kw_rep = NamedSharding(mesh_xy, P())

		@partial(jax.jit,
		         in_shardings=(in_xy, in_xy, kw_rep),
		         out_shardings=out_xy)
		def _compute_gram_q0(P_v: jax.Array, P_c: jax.Array,
		                     kw: jax.Array) -> jax.Array:
			# Per-k product of (conj P_v) × P_c, weighted by kw[k], summed.
			# Broadcasting kw to (nk, 1, 1) matches the (nk, μ, ν) layout.
			prod = jnp.conj(P_v) * P_c
			G = jnp.sum(kw[:, None, None] * prod, axis=0)
			# Symmetrize: the q=0 Gram is Hermitian by construction, but
			# fp-roundoff + reduction-order noise can break it. The
			# pivoted-Cholesky select does its own diagonal clamp, but
			# symmetrizing here costs only O(n_rmu²) and keeps the select
			# on a bit-cleaner input.
			G = 0.5 * (G + jnp.conj(G.T))
			return G

		_isdf_pipeline_cache[cache_key] = _compute_gram_q0

	return _isdf_pipeline_cache[cache_key](P_v_k, P_c_k, k_weights)


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
		P_l_k_muz: (nk, n_rmu, n_zchunk) left pair density, flat-k, P(None, 'x', 'y').
		P_r_k_muz: (nk, n_rmu, n_zchunk) right pair density, flat-k, P(None, 'x', 'y').
		kgrid: (nkx, nky, nkz) — 3-D form appears only inside the FFT helper.
		mesh_xy: Device mesh

	Returns:
		Z_q: (nq, n_rmu, n_zchunk) flat-q, P(None, 'x', 'y').
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, n_zchunk = P_l_k_muz.shape
	assert nk == nkx * nky * nkz, (
		f"P_l_k_muz flat-k dim {nk} does not match kgrid product {nkx*nky*nkz}"
	)
	assert P_r_k_muz.shape == P_l_k_muz.shape, (
		f"P_l/P_r shape mismatch: {P_l_k_muz.shape} vs {P_r_k_muz.shape}"
	)

	cache_key = ('ZCT_LR', id(mesh_xy), nk, n_rmu, n_zchunk)

	if cache_key not in _isdf_pipeline_cache:
		flat_spec = P(None, 'x', 'y')
		flat_shard = NamedSharding(mesh_xy, flat_spec)
		spec_3d = P(None, None, None, 'x', 'y')
		local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, spec_3d, norm='forward')
		local_fftn  = make_flat_k_fftn( mesh_xy, kgrid, spec_3d, norm='forward')

		@partial(jax.jit, in_shardings=flat_shard, out_shardings=flat_shard,
		         donate_argnums=(0,))
		def _left_ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn(P_l))

		@partial(jax.jit,
		         in_shardings=(flat_shard, flat_shard), out_shardings=flat_shard,
		         donate_argnums=(0, 1))
		def _right_ifft_mul_fft(P_r: jax.Array, P_l_Rt: jax.Array) -> jax.Array:
			P_r_Rt = local_ifftn(P_r)
			Z_Rt = P_l_Rt * P_r_Rt
			return local_fftn(Z_Rt)

		def _compute_ZCT_LR(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			P_l_Rt = _left_ifft_conj(P_l)
			return _right_ifft_mul_fft(P_r, P_l_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_ZCT_LR

	return _isdf_pipeline_cache[cache_key](P_l_k_muz, P_r_k_muz)


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
    vertex_mu_L: int = 0,
) -> jax.Array:
    """
    Compute system-matrix L_q from CCT matrix.

    For ``vertex_mu_L == 0`` (standard spin-traced path) the CCT is
    Hermitian positive-definite (modulo numerical noise); we run the
    optimized 2D blocked Cholesky and return the lower-triangular
    factor.  Downstream :func:`solve_zeta_from_L_q` then does two
    triangular solves per-q.

    For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
    the CCT is Hermitian but **indefinite** — Cholesky NaNs and the LU
    fallback in :func:`solve_zeta_from_L_q` is required.  In this case
    we skip the factorization here and pass C_q through unchanged; the
    solve routine consumes it via ``jnp.linalg.solve`` on a per-q-batch
    basis (one LU per call, small enough that explicit
    ``lu_factor`` + ``lu_solve`` reuse buys nothing).

    Note: the artifact return type does NOT change with ``vertex_mu_L``
    — both branches return a (nq, n_rmu, n_rmu) array sharded
    ``P(None, 'x', 'y')``.  Only its semantic interpretation differs
    (Cholesky factor vs system matrix).

    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
        mesh_xy: 2D device mesh
        block_size: Tile block size (auto if None)
        vertex_mu_L: Lorentz vertex index (0 = spin-traced PSD path,
                     1/2/3 = transverse indefinite path).

    Returns:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor (μ_L=0) or unmodified
             C_q (μ_L=1,2,3), sharded P(None, 'x', 'y')
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"

    # Indefinite-CCT path: skip the Cholesky outright.  ``solve_zeta_from_L_q``
    # consumes C_q directly via ``jnp.linalg.solve`` (LU + back-solve internally).
    if int(vertex_mu_L) != 0:
        L_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        return jax.lax.with_sharding_constraint(C_q, L_shard)

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
    vertex_mu_L: int = 0,
) -> jax.Array:
    """
    Solve for zeta_q given pre-computed system matrix from
    :func:`compute_L_q_from_CCT`.

    For ``vertex_mu_L == 0`` ``L_q`` is the lower-triangular Cholesky
    factor of CCT and the inner solve is two triangular substitutions
    (``L y = Z`` then ``L^H ζ = y``).  This is the historical fast
    path — bit-identical to the previous implementation.

    For ``vertex_mu_L != 0`` ``L_q`` is the *unfactored* CCT matrix
    (transverse-channel γ̃^i CCT is Hermitian but indefinite, so
    Cholesky is invalid).  The inner solve dispatches to
    ``jnp.linalg.solve`` which performs an LU decomposition with
    partial pivoting and a back-solve in one shot.  Per-q matrices are
    small enough that explicit ``lu_factor`` + ``lu_solve`` reuse
    buys nothing — both batched ``jnp.linalg.solve`` and the
    triangular path see the same shard_map'd, q-chunked, batched-vmap
    structure with identical input/output shardings.

    Uses q-chunked all-gather strategy: gather B_q matrices at a time,
    then solve all B_q systems in parallel using vmap.

    Memory trade-off:
    - q_chunk_size=1: Minimum memory (one matrix replicated at a time)
    - q_chunk_size=nq: Maximum parallelism (all matrices replicated)

    Args:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor (μ_L=0) or raw CCT
             (μ_L=1,2,3), sharded P(None, 'x', 'y')
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')
             or P(None, None, ('x','y')) if caller already resharded
        mesh_xy: 2D device mesh
        q_chunk_size: Number of q-points to solve simultaneously (default 1)
        vertex_mu_L: Lorentz vertex index — selects Cholesky-back-solve
                     vs jnp.linalg.solve.  Output sharding is identical
                     in both branches.

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

    use_lu = (int(vertex_mu_L) != 0)

    # Cache key for solve function (includes q_chunk_size and padded size).
    # ``use_lu`` partitions the cache so the Cholesky and LU compiles
    # don't collide on the same key.
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu,
                 n_zchunk_padded, q_chunk_size, bool(use_lu))

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            if use_lu:
                # Indefinite CCT: jnp.linalg.solve does LU + back-solve.
                return jnp.linalg.solve(L, Z_cols)
            y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
            zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return zeta

        # Vectorized solve for a batch of q-points
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ('x', 'y'))),
                 out_specs=P(None, None, ('x', 'y')))
        def _sharded_cho_solve_batch(L_batch: jax.Array, Z_batch: jax.Array) -> jax.Array:
            """Solve (B_q, n_rmu, n_rmu) @ (B_q, n_rmu, n_cols) -> (B_q, n_rmu, n_cols)"""
            if use_lu:
                # jnp.linalg.solve is natively batched on the leading axis;
                # JAX dispatches one LU factorization per q internally.
                return jnp.linalg.solve(L_batch, Z_batch)
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
        # Donation on Z_q is the key — the caller `del Z_q` immediately
        # after this block, so XLA can alias the input buffer for the
        # output (2× tile theoretical minimum).  Without donation SPMD
        # hits an Involuntary full rematerialization going from
        # P(None,'x','y') → P(None,None,('x','y')) because both mesh
        # axes need to re-shard at once.  Measured at Si 4×4×4 60Ry
        # (nq=64, μ=2400, B_r=12672, 2×2 mesh):
        #   direct, no donate: 31.14 GB/dev (temp 15.57 GB, Involuntary Remat)
        #   direct, donate:    15.57 GB/dev (temp 7.79 GB) -- 50% reduction
        @partial(jax.jit, donate_argnums=(0,))
        def _reshard_z(z):
            return jax.lax.with_sharding_constraint(z, _target_sharding)
        Z_col = _reshard_z(Z_q)
        # No-op when called inside an outer jit (tracer has no
        # block_until_ready and the outer jit syncs at its boundary).
        if not isinstance(Z_col, jax.core.Tracer):
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


# =============================================================================
# Jittable r-chunk body — the whole per-r-chunk iteration in one jit
# =============================================================================
#
# This is the hot kernel: load-phase FFT → reshard → streaming pair-density
# accumulate (both L and R) over all band-chunks → ZCT → reshard → solve,
# all fused into one jax.jit.  The Python driver only does persistent-state
# setup (centroids, L_q, G-space cache) and H5 I/O writes around the jit call.
#
# All "structural" configuration (band_chunk_ranges, band_range_left/right,
# actual_n_rchunk, q_chunk_size, kvecs_frac, mesh, meta) is closure state —
# the factory compiles a distinct jit per (hashable) tuple.  The typical
# r-chunk loop has exactly TWO compiled variants: the full-sized r-chunks
# and the last remainder.
#
# Dynamic inputs: the pre-loaded G-space tuple (one array per band-chunk),
# the centroid copies and L_q (persistent across the full fit), band_norms
# (normalised to jnp.ones(nb) when absent so the shape is uniform), and
# r_start as a scalar dynamic int.

_fit_one_rchunk_cache: dict = {}


def _make_fit_one_rchunk_kernel(
    mesh_xy: Mesh,
    meta: Meta,
    band_chunk_ranges: tuple[tuple[int, int], ...],
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_range_full: tuple[int, int],
    actual_n_rchunk: int,
    q_chunk_size: int,
    kvecs_frac,
    psi_G_store,
    vertex_mu_L: int = 0,
):
    """Factory: returns a ``jax.jit``'d fit_one_rchunk callable closing
    over every piece of static structure + a :class:`PsiGStore` that
    supplies per-bc ψ(G) slices from host memory.

    Returned function signature::

        zeta_chunk = kernel(
            psi_l_rmuT_X_fit,
            psi_r_rmuT_X_fit,
            L_q,
            norms_l, norms_r,      # jax arrays; jnp.ones when no band_norms
            r_start_dyn,           # scalar int32
        )

    ψ(G) is NOT a jit argument.  The bc-loop fetches each chunk from
    the host-resident store via ``io_callback`` — only the currently
    active bc lives on device.  See :mod:`common.psi_G_store` for
    lifecycle + layout details.

    The inner body fully composes the load-phase FFT + reshard, the
    band-chunk pair-density streaming loop (Python-unrolled at trace),
    the ZCT, the Z_q→Z_col reshard, and the Cholesky solve.
    """
    from .load_wfns import get_sharded_wfns_rchunk_slice

    nk_tot = meta.nk_tot
    n_rmu = meta.n_rmu
    nspinor = meta.nspinor
    kgrid = meta.kgrid

    l_lo, l_hi = band_range_left
    r_lo, r_hi = band_range_right

    P_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_slice_shard = NamedSharding(mesh_xy, P(None, 'x', None, None))

    # Classify each band-chunk against the L and R endpoint ranges at
    # trace time.  Status codes (``'skip' | 'direct' | 'pad'``) dispatch
    # the Python if-branch inside _kernel below so the jit body only
    # emits the ops that matter for this bc.
    #
    #   skip   — bc_range is entirely outside [rs, re); the contribution
    #            is dropped at trace (no ops emitted).
    #   direct — bc_range ⊆ [rs, re); straight slice of the centroid
    #            copy, no zero-padding needed.
    #   pad    — bc_range straddles an endpoint; slot the overlapping
    #            bands into a zero-padded bc_size-wide buffer so the
    #            pair-density einsum still hits the uniform jit cache.
    def _classify(rs, re, bc_lo, bc_hi):
        ol_lo = max(bc_lo, rs)
        ol_hi = min(bc_hi, re)
        if ol_hi <= ol_lo:
            return 'skip', None
        if ol_lo == bc_lo and ol_hi == bc_hi:
            return 'direct', (ol_lo - rs, ol_hi - rs)
        return 'pad', (ol_lo - rs, ol_hi - rs, ol_lo - bc_lo, ol_hi - bc_lo)

    bc_classify = [
        (
            bc_range,
            _classify(l_lo, l_hi, bc_range[0], bc_range[1]),
            _classify(r_lo, r_hi, bc_range[0], bc_range[1]),
        )
        for bc_range in band_chunk_ranges
    ]

    @jax.jit
    def _kernel(
        psi_l_rmuT_X_fit,
        psi_r_rmuT_X_fit,
        L_q,
        norms_l,
        norms_r,
        r_start_dyn,
    ):
        # --- 1. Pair-density accumulators (one r-chunk wide) ---
        def _zero_P():
            return jax.lax.with_sharding_constraint(
                jnp.zeros((nk_tot, n_rmu, actual_n_rchunk),
                          dtype=jnp.complex128),
                P_sharding)
        P_l = _zero_P()
        P_r = _zero_P()

        # Pair-density bc-contribution — shared for the L and R sides.
        # ``cls`` is the _classify tuple for THIS side; ``psi_centroid``
        # is psi_l_rmuT_X_fit (for L) or psi_r_rmuT_X_fit (for R);
        # ``norms`` is the matching clamped weights.  Returns the
        # updated accumulator (P_l or P_r).
        def _accumulate(P_acc, cls, psi_centroid, norms, psi_bc_Y, bc_size):
            tag, payload = cls
            if tag == 'skip':
                return P_acc
            if tag == 'direct':
                lo, hi = payload
                psi_L_bc = psi_centroid[:, :, lo:hi, :]
                norm_slice = norms[lo:hi]
            else:  # 'pad' — straddles an endpoint
                cen_lo, cen_hi, dst_lo, dst_hi = payload
                psi_L_bc = jax.lax.with_sharding_constraint(
                    jnp.zeros((nk_tot, n_rmu, bc_size, nspinor),
                              dtype=jnp.complex128),
                    L_slice_shard)
                psi_L_bc = psi_L_bc.at[:, :, dst_lo:dst_hi, :].set(
                    psi_centroid[:, :, cen_lo:cen_hi, :])
                norm_slice = jnp.ones((bc_size,), dtype=jnp.float64
                                      ).at[dst_lo:dst_hi].set(
                    norms[cen_lo:cen_hi])
            psi_R_bc = psi_bc_Y / norm_slice[None, :, None, None]
            if vertex_mu_L == 0:
                return accumulate_pair_density_spin_traced(
                    P_acc, psi_L_bc, psi_R_bc, mesh_xy)
            return accumulate_pair_density_lorentz(
                P_acc, psi_L_bc, psi_R_bc, vertex_mu_L, mesh_xy)

        # --- 2. Stream band-chunks: fetch ψ(G) per-bc from host via
        #       io_callback, FFT + reshard, accumulate pair density.
        #       Each bc's ψ(G) is live only during its own iteration.
        for bc_idx, (bc_range, l_cls, r_cls) in enumerate(bc_classify):
            bc_size = bc_range[1] - bc_range[0]
            psi_bc_G = psi_G_store.fetch_psi_G(bc_range)
            psi_bc_Y = get_sharded_wfns_rchunk_slice(
                psi_bc_G, meta, r_start_dyn, actual_n_rchunk,
                kvecs_frac, mesh_xy, bc_range)
            P_l = _accumulate(P_l, l_cls, psi_l_rmuT_X_fit,
                              norms_l, psi_bc_Y, bc_size)
            P_r = _accumulate(P_r, r_cls, psi_r_rmuT_X_fit,
                              norms_r, psi_bc_Y, bc_size)

        # 3. ZCT + reshard + solve
        Z_q = compute_ZCT_from_left_right_zchunk(P_l, P_r, kgrid, mesh_xy)
        z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
        Z_col = jax.lax.with_sharding_constraint(Z_q, z_col_shard)
        zeta = solve_zeta_from_L_q(
            L_q, Z_col, mesh_xy, q_chunk_size, vertex_mu_L=vertex_mu_L)
        return zeta

    return _kernel


def fit_one_rchunk(
    *,
    psi_G_store,
    psi_l_rmuT_X_fit,
    psi_r_rmuT_X_fit,
    L_q,
    norms_l,
    norms_r,
    r_start_dyn,
    mesh_xy: Mesh,
    meta: Meta,
    band_chunk_ranges: tuple[tuple[int, int], ...],
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_range_full: tuple[int, int],
    actual_n_rchunk: int,
    q_chunk_size: int,
    kvecs_frac: np.ndarray,
    vertex_mu_L: int = 0,
):
    """Entry point for the r-chunk body jit.  Caches one compiled kernel
    per distinct static configuration.

    ``psi_G_store`` is captured in the jit closure (not a jit arg) so
    the compiled kernel calls its ``fetch_psi_G`` method via io_callback
    inside the bc-loop.  The cache key includes ``id(psi_G_store)`` to
    avoid reusing a compile built against a different store.
    """
    cache_key = (
        id(mesh_xy),
        actual_n_rchunk,
        tuple(tuple(b) for b in band_chunk_ranges),
        tuple(band_range_left), tuple(band_range_right),
        tuple(band_range_full),
        q_chunk_size,
        meta.n_rmu, meta.nk_tot, meta.nspinor,
        tuple(meta.fft_grid),
        hash(kvecs_frac.tobytes()),
        id(psi_G_store),
        int(vertex_mu_L),
    )
    fn = _fit_one_rchunk_cache.get(cache_key)
    if fn is None:
        fn = _make_fit_one_rchunk_kernel(
            mesh_xy, meta,
            tuple(tuple(b) for b in band_chunk_ranges),
            tuple(band_range_left),
            tuple(band_range_right),
            tuple(band_range_full),
            actual_n_rchunk,
            q_chunk_size,
            kvecs_frac,
            psi_G_store,
            vertex_mu_L=int(vertex_mu_L),
        )
        _fit_one_rchunk_cache[cache_key] = fn
    return fn(
        psi_l_rmuT_X_fit,
        psi_r_rmuT_X_fit,
        L_q,
        norms_l,
        norms_r,
        r_start_dyn,
    )


def _band_norms_slice(
    band_norms: np.ndarray | None, band_range: tuple[int, int], nb: int,
) -> jax.Array:
    """Slice + clamp the pseudobands weights to a ``(nb,)`` jax array.

    Divisor is ``max(1, w_n)``: low-weight pseudobands keep their
    sub-unit norm (DOS-preserving), high-weight ones are pulled back
    to unit (no dominance), zero-weight windows stay at 1.0 since the
    ``max(1, 0)=1`` floor avoids a divide-by-zero.  When
    ``band_norms`` is ``None`` (no pseudobands), returns ``jnp.ones``.
    """
    if band_norms is None:
        return jnp.ones((nb,), dtype=jnp.float64)
    lo, hi = band_range
    return jnp.maximum(jnp.asarray(band_norms[lo:hi], dtype=jnp.float64), 1.0)


# ψ(G) no longer enters the jit as a tuple of arguments.  It lives on
# host, sharded n_XY over bands in per-rank tiles — each rank owns one
# contiguous ``(nk, nb/P, ns, nx, ny, nz)`` numpy array — and the
# kernel body slices per-bc (and optionally per-k) subsets via
# :func:`common.psi_G_store.PsiGStore.fetch_psi_G`.  See that module
# for the two lifecycle modes (host_cache vs file_reread).


def fit_zeta_chunked_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    chunk_r: int,
    output_file: str,
    psi_rmu_Y: jax.Array,
    psi_rmuT_X: jax.Array,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    k_chunk_size: int = 0,
    q_gather_size: int = 0,
    band_norms: np.ndarray | None = None,
    use_ffi_io: bool = False,
    gspace_mode: str = "host_cache",
    vertex_mu_L: int = 0,
    max_r_chunks: int = -1,
):
    """
    Full zeta fitting pipeline with r-chunk loop and HDF5 output.

    Workflow:
    1. Slice pre-loaded centroid wavefunctions into left/right halves.
    2. Compute C_q from left/right pair density via FFT.
    3. Compute L_q = chol(C_q) using 2D blocked algorithm.
    4. For each r-chunk:
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
        psi_rmu_Y:  Centroid wavefunctions for the full [b0, b4) band range,
                    shape (nk, nb_full, ns, n_rmu), P(None, None, None, 'y'),
                    un-conjugated ψ.  Produced by
                    :func:`common.load_wfns.load_centroids_band_chunked`.
        psi_rmuT_X: Same centroid data transposed/sharded for the pair-density
                    kernel, shape (nk, n_rmu, nb_full, ns),
                    P(None, 'x', None, None), conjugated ψ*.
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        gspace_mode: ``"host_cache"`` (default) loads all ψ(G) band-chunks
                     into host RAM once at startup and pulls per-bc shards
                     into the jit via io_callback.  ``"file_reread"`` drops
                     the host cache between r-chunks and re-reads via
                     phdf5 collective I/O.  In both modes the jit never
                     holds more than one bc's ψ(G) on device.
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)

    Returns:
        peak_bytes:  GPU high-water mark (peak_bytes_in_use) during chunk loop

    The centroid wavefunctions are inputs, not outputs — the caller is
    expected to hold the single ``load_centroids_band_chunked`` result and
    reuse it for :func:`gw.wavefunction_bundle.build_wavefunctions` after
    the fit completes.
    """
    import h5py

    nx, ny, nz = meta.fft_grid
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

    print(f"\n  Zeta fitting: {num_chunks} r-chunks x {n_rchunk} r-points, "
          f"{nb_full} bands ({nb_left} left + {nb_right} right)")
    print(f"  Output: {output_file}")

    # ========== STEP 1: Slice pre-loaded centroid ψ into left/right halves ==========
    with timing.section("zeta_fit.slice_halves"):
        # Band range arithmetic — left/right are sub-ranges of [b0, b4).
        l_band_start = band_range_left[0] - band_range_full[0]
        l_band_end = l_band_start + nb_left
        r_band_start = band_range_right[0] - band_range_full[0]
        r_band_end = r_band_start + nb_right

        # Cheap views — the caller keeps the full arrays alive for the
        # post-fit wfn bundle build, so we don't need independent copies.
        psi_l_rmu_Y = psi_rmu_Y[:, l_band_start:l_band_end, :, :]
        psi_l_rmuT_X = psi_rmuT_X[:, :, l_band_start:l_band_end, :]
        psi_r_rmu_Y = psi_rmu_Y[:, r_band_start:r_band_end, :, :]
        psi_r_rmuT_X = psi_rmuT_X[:, :, r_band_start:r_band_end, :]

        print(f"  Left wfns:  {psi_l_rmu_Y.shape}")
        print(f"  Right wfns: {psi_r_rmu_Y.shape}")

        # Pseudobands: clamp weights to ``max(1, w_n)`` and apply them to
        # the centroid copies used for CCT.  When band_norms is None the
        # slices are jnp.ones → the *_fit aliases are identical to the
        # *_rmu_Y / *_rmuT_X copies.  See _band_norms_slice for the why.
        norms_l_jax = _band_norms_slice(band_norms, band_range_left, nb_left)
        norms_r_jax = _band_norms_slice(band_norms, band_range_right, nb_right)
        # psi shapes: Y=(nk, nb, ns, n_rmu), X=(nk, n_rmu, nb, ns)
        psi_l_rmu_Y_fit = psi_l_rmu_Y / norms_l_jax[None, :, None, None]
        psi_l_rmuT_X_fit = psi_l_rmuT_X / norms_l_jax[None, None, :, None]
        psi_r_rmu_Y_fit = psi_r_rmu_Y / norms_r_jax[None, :, None, None]
        psi_r_rmuT_X_fit = psi_r_rmuT_X / norms_r_jax[None, None, :, None]
        if band_norms is not None:
            n_weighted = int(np.sum(band_norms > 1.01))
            n_zero = int(np.sum(band_norms < 1e-10))
            print(f"  Pseudobands normalization: {n_weighted} weighted, "
                  f"{n_zero} zero-weight (skipped)")

    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # Uses normalized copies for fitting (equal-weight pair densities)
    # γ̃^0 = I_4, so vertex_mu_L=0 is the standard spin-traced path; non-zero
    # μ_L swaps in compute_pair_density_lorentz (ns=4 only).
    with timing.section("zeta_fit.CCT"):
        if vertex_mu_L == 0:
            print(f"  Computing pair densities P_l, P_r (spin_traced)")
            P_l_k = compute_pair_density_spin_traced(psi_l_rmuT_X_fit, psi_l_rmu_Y_fit, mesh_xy)
            P_r_k = compute_pair_density_spin_traced(psi_r_rmuT_X_fit, psi_r_rmu_Y_fit, mesh_xy)
        else:
            print(f"  Computing pair densities P_l, P_r (γ̃^{vertex_mu_L} vertex)")
            P_l_k = compute_pair_density_lorentz(psi_l_rmuT_X_fit, psi_l_rmu_Y_fit, vertex_mu_L, mesh_xy)
            P_r_k = compute_pair_density_lorentz(psi_r_rmuT_X_fit, psi_r_rmu_Y_fit, vertex_mu_L, mesh_xy)
        P_l_k.block_until_ready()
        P_r_k.block_until_ready()
        C_q = compute_CCT_from_left_right(P_l_k, P_r_k, kgrid, mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu, n_rmu)

        # Free pair densities - only needed for C_q
        del P_l_k, P_r_k

        # Flatten for Cholesky
        C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

    # ========== STEP 3: Compute L_q (factor for μ_L=0; raw CCT for μ_L=i) ==========
    with timing.section("zeta_fit.cholesky"):
        if int(vertex_mu_L) == 0:
            print(f"  Computing L_q = chol(C_q)")
        else:
            print(f"  Skipping Cholesky (vertex_mu_L={vertex_mu_L} CCT is "
                  f"indefinite); will LU-solve per q-batch")
        L_q = compute_L_q_from_CCT(
            C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L))
        L_q.block_until_ready()
        print(f"  L_q: {L_q.shape}")

    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        gc.collect()
        jax.clear_caches()  # Clear JAX function caches that may hold array refs

    # ========== STEP 4: Create HDF5 file ==========
    # zeta_q is stored flat-q: shape (nq, n_rmu, n_rtot) with
    # q_flat = qx*nqy*nqz + qy*nqz + qz.  Flat-q is the ongoing
    # convention across LORRAX; see file_io.slab_io docs.  Chunk by
    # single-q r-slice so per-q reads stay contiguous.
    #
    # Single SlabIO handle reused for both create_dataset and all
    # writes — avoids the ~900 ms cost of a second collective
    # H5Fopen/close pair (measured 2026-04-18 at MoS2 3x3).  The
    # allgather backend doesn't need a long-lived handle (rank 0 writes
    # from a Python worker using plain h5py) so we keep the old
    # create-then-reopen pattern for that path.
    from file_io.slab_io import SlabIO
    # Dataset layout ``(nq, n_rtot, n_rmu)`` — NOT ``(nq, n_rmu, n_rtot)``.
    # Rationale: per-r-chunk writes span the full innermost axis (n_rmu)
    # under this layout, so each ``(q, r)`` row is contiguous on disk.
    # Under the old ``(nq, n_rmu, n_rtot)`` layout we'd write n_rchunk <
    # n_rtot on the innermost axis, producing 480K × 1920-B scattered
    # strips per rank per write (measured at 0.18 GB/s on Perlmutter
    # pscratch, 8× slower than contiguous).  Per-q reads (V_q) stay
    # contiguous under this layout too: a 6.6 M-element slab at
    # ``(q, 0, 0)`` is a single contiguous block.  Downstream V_q
    # transposes the returned array on GPU to match the kernel's
    # (n_rmu, n_rtot) expectation — ~50 µs per q, negligible.
    with timing.section("zeta_fit.open_file"):
        if use_ffi_io:
            zeta_io = SlabIO(output_file, mode='w', mesh=mesh_xy,
                             use_ffi_io=True)
            zeta_io.create_dataset(
                'zeta_q',
                shape=(nq, n_rtot, n_rmu),
                dtype=np.complex128,
                chunks=(1, n_rchunk, n_rmu),
            )
        else:
            with SlabIO(output_file, mode='w', mesh=mesh_xy,
                        use_ffi_io=False) as _zeta_create_io:
                _zeta_create_io.create_dataset(
                    'zeta_q',
                    shape=(nq, n_rtot, n_rmu),
                    dtype=np.complex128,
                    chunks=(1, n_rchunk, n_rmu),
                )
            zeta_io = None

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # Env-var override: forces the slow path (re-read WFN.h5 per
    # r-chunk) even when memory would allow host caching.  Useful for
    # probing the scaling regime where wavefunctions don't fit in host
    # memory (multi-TB WFN.h5).  LORRAX_GSPACE_MODE=file_reread to
    # override the caller's default.
    _gspace_mode_override = os.environ.get("LORRAX_GSPACE_MODE")
    if _gspace_mode_override:
        gspace_mode = _gspace_mode_override

    # Uniform band chunks over [b_full_start, b_full_end]: N-1 of
    # size ``band_chunk_size`` plus one remainder chunk.  This gives
    # the read/FFT pipeline and the pair-density einsum exactly
    # TWO compile shapes, regardless of where the L/R endpoints fall.
    # Chunks that straddle an L/R endpoint get handled in the loop
    # below by padding the left-side ``psi_L_bc`` slice with zero
    # bands — the resulting einsum still runs at the uniform
    # ``bc_size``, so it hits the same JIT cache.
    _bfs, _bfe = band_range_full
    band_chunk_ranges = [
        (_bfs + i * band_chunk_size,
         min(_bfs + (i + 1) * band_chunk_size, _bfe))
        for i in range((_bfe - _bfs + band_chunk_size - 1) // band_chunk_size)
    ]

    # Build the host-resident ψ(G) store.  Both modes keep zero
    # persistent device residency — the jit fetches one bc at a time
    # via io_callback.  See :mod:`common.psi_G_store` for details.
    from common.psi_G_store import build_psi_G_store
    psi_G_store = build_psi_G_store(
        wfn=wfn, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        mode=gspace_mode,
    )

    # ========== STEP 6: Loop over chunks ==========
    # Wall-clock totals for the end-of-fit timing line.  ``t_fit_total``
    # covers the fused fit_one_rchunk jit (load + pair + ZCT + solve) —
    # finer-grained breakdown now lives inside the jit and is only
    # observable via xprof, not perf_counter.
    t_fit_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()

    # Per-chunk writes go through zeta_io (SlabIO).  On the allgather
    # backend we keep the old async-writer pattern: main thread does
    # the allgather, rank-0 background thread does the h5py hyperslab
    # writes — this hides the h5 latency behind the next chunk's GPU
    # compute.  On the FFI backend, writes are collective so all ranks
    # must enter in lock-step; the synchronous SlabIO.write_slab call
    # from the main thread is the right shape (H5Dwrite's host-block
    # doesn't stall the CUDA stream, so the next chunk's build still
    # overlaps).
    write_queue = queue.Queue()
    write_error = [None]

    def writer_worker():
        """Rank-0 background thread: dequeue + h5py hyperslab writes."""
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                zeta_data, r_start, r_end, chunk_id, q_start, q_end = item
                # zeta_data is already a host numpy array of shape
                # (q_end-q_start, n_rmu, r_end-r_start) in the order
                # the shard_map produced.  Dataset on disk is
                # (nq, n_rtot, n_rmu) — swap the last two axes at
                # write time.  h5py is happy with a non-contiguous
                # source (stride-swap view); it linearizes internally.
                with h5py.File(output_file, 'a') as f:
                    for i, q_flat in enumerate(range(q_start, q_end)):
                        # zeta_data[i] is (n_rmu, r_chunk) → transpose
                        # to (r_chunk, n_rmu) to match file layout.
                        f['zeta_q'][q_flat, r_start:r_end, :] = zeta_data[i].T
                write_queue.task_done()
        except Exception as e:
            write_error[0] = e
            write_queue.task_done()

    writer_thread = None
    if not use_ffi_io and jax.process_index() == 0:
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()

    # Peak GPU memory tracker — reports the all-time high-water mark (peak_bytes_in_use).
    # This is the number that determines whether you OOM: it includes JIT caches and
    # prior-stage allocations, not just the chunk loop arrays.
    # GPU high-water tracker.  The JAX CUDA PJRT on this stack returns
    # None from ``memory_stats()``, so we fall back to a single
    # nvidia-smi sample at the end of the chunk loop.  Sampled once
    # (not per-chunk) because concurrent nvidia-smi from all 4 ranks
    # inside the Shifter container has been observed to hang on some
    # Perlmutter node types.
    _peak_bytes = 0
    def _track_peak():
        nonlocal _peak_bytes
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,nounits,noheader", "--id=0"],
                text=True, timeout=2).strip()
            mb = int(out.splitlines()[0])
            _peak_bytes = max(_peak_bytes, mb * (1024 ** 2))
        except Exception:
            pass  # leave _peak_bytes = 0; caller suppresses the print

    from common.progress import LoopProgress
    r_progress = LoopProgress(
        num_chunks, print, title="zeta fitting",
        item_name="r-chunk", max_updates=min(num_chunks, 20))

    # norms_l_jax / norms_r_jax were built in STEP 1 above — reuse them
    # as the uniform-shape (nb,) inputs to the fit_one_rchunk jit.

    with timing.section("zeta_fit.chunk_loop"):
        chunks_to_run = num_chunks if max_r_chunks <= 0 else min(num_chunks, max_r_chunks)
        for chunk_idx in range(chunks_to_run):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start

            # file_reread mode: (re)build the host-side ψ(G) tiles
            # for this r-chunk.  host_cache mode: no-op.
            psi_G_store.begin_rchunk(r_start, r_end)

            t0 = time.perf_counter()
            try:
                with timing.section("zeta_fit.chunk.fit_one_rchunk"), \
                     jax_profile.step_annotation("chunk_fit", step_num=chunk_idx):
                    zeta_chunk = fit_one_rchunk(
                        psi_G_store=psi_G_store,
                        psi_l_rmuT_X_fit=psi_l_rmuT_X_fit,
                        psi_r_rmuT_X_fit=psi_r_rmuT_X_fit,
                        L_q=L_q,
                        norms_l=norms_l_jax,
                        norms_r=norms_r_jax,
                        r_start_dyn=jnp.asarray(r_start, dtype=jnp.int32),
                        mesh_xy=mesh_xy,
                        meta=meta,
                        band_chunk_ranges=band_chunk_ranges,
                        band_range_left=band_range_left,
                        band_range_right=band_range_right,
                        band_range_full=band_range_full,
                        actual_n_rchunk=actual_n_rchunk,
                        q_chunk_size=q_chunk_size,
                        kvecs_frac=kvecs_frac,
                        vertex_mu_L=vertex_mu_L,
                    )
                    zeta_chunk.block_until_ready()
            finally:
                # MUST run after block_until_ready — under file_reread
                # the host tiles are freed here and any still-pending
                # io_callback would use-after-free.
                psi_G_store.end_rchunk()
            t_fit_total += time.perf_counter() - t0

            # 6e. Q-chunked allgather → host copy → async HDF5 write.
            # The allgather replicates zeta slices: per-device output is
            # (q_gather, n_rmu, chunk_r) which at large chunk_r can be huge.
            # Chunking over q keeps each allgather under memory limits.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                # Each allgather produces a FULLY REPLICATED output per device:
                # q_gather × n_rmu × chunk_r × 16 bytes, plus NCCL temp of same size.
                # Cap to keep replicated output + NCCL under available memory.
                _bytes_per_q_replicated = 2 * n_rmu * actual_n_rchunk * 16
                _safe_q_gather = max(1, min(nq, int(10 * 1024**3 / max(1, _bytes_per_q_replicated))))
                if q_gather_size > 0:
                    _q_gather = min(nq, q_gather_size, _safe_q_gather)
                else:
                    _q_gather = _safe_q_gather

                if use_ffi_io:
                    # FFI path: zeta_chunk is (nq, n_rmu, chunk_r),
                    # dataset is (nq, n_rtot, n_rmu) — transpose the
                    # last two axes before writing so the slab matches
                    # the disk layout.  The transpose is a JAX
                    # metadata-only operation (shard axis 'chunk_r' /
                    # sharded → axis 1 of the post-transpose tensor;
                    # NamedSharding updates in place, no data motion).
                    # Per-rank write pattern goes from ~120 K small
                    # strips (old (nq, n_rmu, n_rtot) layout) to 1000
                    # fat contiguous strips (one per q, full n_rmu +
                    # rank's n_rchunk/4 rows).
                    zeta_chunk_write = zeta_chunk.transpose(0, 2, 1)
                    zeta_io.write_slab(
                        'zeta_q', zeta_chunk_write,
                        offset=(0, r_start, 0),
                        global_shape=(nq, n_rtot, n_rmu))
                else:
                    # Allgather path: gather once per q-chunk on every rank,
                    # queue the per-q hyperslab writes on rank 0's
                    # background thread so the h5py I/O is hidden behind
                    # the next chunk's GPU compute.  process_allgather is
                    # itself a blocking D2H+collective — there is no async
                    # allgather-to-host API in JAX today — so the main
                    # thread stall per chunk is roughly (cross-rank NCCL
                    # time) + (PCIe D2H).  First chunk eats an extra ~1 s
                    # of NCCL/XLA first-collective setup.
                    for _q0 in range(0, nq, _q_gather):
                        _q1 = min(_q0 + _q_gather, nq)
                        _slice = zeta_chunk[_q0:_q1]
                        _gathered = jax.experimental.multihost_utils.process_allgather(
                            _slice, tiled=False)
                        if jax.process_index() == 0:
                            _g = np.asarray(_gathered)
                            if _g.ndim == 4 and _g.shape[0] == 1:
                                _g = _g[0]
                            write_queue.put((_g, r_start, r_end, chunk_idx, _q0, _q1))
                        del _gathered, _slice

                # No per-chunk sync_global_devices here: the
                # allgather is itself a collective so all ranks are
                # already aligned at this point, and we want the main
                # thread free to start next chunk's GPU compute while
                # rank 0's writer thread flushes to disk.  Final
                # sync_global_devices("zeta_writes_complete") at the
                # bottom of this function serves as the rendezvous.
                del zeta_chunk
            t_write_total += time.perf_counter() - t0
            r_progress.step()


    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    # Sample GPU memory ONCE after the last chunk's jit settles.  The
    # allocator keeps the peak reservation so this reads close to the
    # all-time high water.
    _track_peak()

    # Drain the rank-0 writer queue (allgather backend only; FFI path
    # writes are already fully flushed by the synchronous SlabIO calls).
    if jax.process_index() == 0 and writer_thread is not None:
        write_queue.join()
        write_queue.put(None)
        writer_thread.join()
        if write_error[0] is not None:
            raise RuntimeError(f"Async writer failed: {write_error[0]}")

    # Close the SlabIO handle (FFI path only; allgather path never
    # opened one after STEP 4).
    with timing.section("zeta_fit.close_io"):
        if zeta_io is not None:
            zeta_io.close()

    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Free the host tiles (host_cache mode only; file_reread's tiles
    # are already empty after the final end_rchunk).  The phdf5 reader
    # itself is cached at module level and survives.
    psi_G_store.close()

    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    print(f"  Zeta output: {output_file}  shape: ({nqx},{nqy},{nqz},{n_rmu},{n_rtot})")
    print(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

    # Return only peak-memory high-water mark; centroid wavefunctions
    # are not returned (see docstring — callers re-load them directly
    # via ``load_centroids_band_chunked``).
    return _peak_bytes

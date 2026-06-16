import gc
import os
import subprocess
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
from .gamma_matrices import (
    gamma_perm_phase as _gamma_perm_phase_mu,
    gamma_double_contract,
)

_MEM_PROFILE = bool(os.environ.get("LORRAX_MEM_PROFILE", ""))

# Running max of nvidia-smi used MB across all probe points within a run
# (this rank's GPU only).  jax.device_memory_stats() returns None on the
# JAX 0.8 / CUDA 12.9 Perlmutter stack, so nvidia-smi is the only way to
# observe the TRUE per-rank HBM peak including cuFFT plan workspace,
# NCCL collective buffers, and other XLA-arena-external allocations.
_NVSMI_PEAK_MB = 0
_NVSMI_LAST_MB = 0

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


def mem_probe(label, *, only_rank0=True):
    """``LORRAX_MEM_DEBUG=1`` runtime probe of process-wide HBM at named sites.

    Reports the JAX/XLA allocator ``bytes_in_use+peak`` plus the top-10
    ``jax.live_arrays()`` shapes.  Module-level so both ``fit_zeta_to_h5``
    (r-chunk loop) and ``gw_init.prepare_isdf_and_wavefunctions`` (V_q
    sites) call the SAME helper — single source of truth for the full
    ζ-fit + V_q HBM lifecycle map.  HLO buffer-assignment.txt is per-jit
    and cannot prove cross-jit liveness; this fills the gap.  Cheap when
    unset (env-var check only; no JAX calls in the early-exit path).

    Round-0 (commit 5c884ac) wired this at three points per r-chunk in
    fit_zeta_to_h5; Round-1 extends to zeta_fit_start, pre_rchunk_loop,
    zeta_fit_end, pre_v_q, post_v_q for the full lifecycle.  Round-7
    (faithfulness audit) adds the ``nvidia-smi`` per-rank true-HBM
    sample — the *canonical* OOM-relevance metric since
    ``device.memory_stats()`` returns ``None`` on this stack.
    """
    if not os.environ.get("LORRAX_MEM_DEBUG"):
        return
    if only_rank0 and jax.process_index() != 0:
        return
    dev = jax.devices()[0]
    stats = dev.memory_stats() if hasattr(dev, "memory_stats") else {}
    if stats is None:
        stats = {}
    bytes_in_use = stats.get("bytes_in_use", -1)
    peak_bytes_in_use = stats.get("peak_bytes_in_use", -1)
    live = jax.live_arrays()
    by_shape = {}
    total_live = 0
    for arr in live:
        if not hasattr(arr, "shape"):
            continue
        try:
            sz = int(np.prod(arr.shape)) * arr.dtype.itemsize
        except Exception:
            continue
        total_live += sz
        key = (tuple(arr.shape), str(arr.dtype))
        entry = by_shape.get(key)
        if entry is None:
            by_shape[key] = [1, sz]
        else:
            entry[0] += 1
            entry[1] += sz
    nvsmi_mb = _nvsmi_used_mb_local_gpu()
    print(f"[mem_probe {label}] in_use={bytes_in_use/1e9:.2f} GB  "
          f"peak={peak_bytes_in_use/1e9:.2f} GB  "
          f"live_count={len(live)} live_total={total_live/1e9:.2f} GB  "
          f"nvsmi={nvsmi_mb/1024:.2f} GB nvsmi_peak={_NVSMI_PEAK_MB/1024:.2f} GB",
          flush=True)
    top = sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:10]
    for (shape, dtype), (cnt, sz) in top:
        print(f"[mem_probe {label}]   {dtype} {shape} x {cnt} = "
              f"{sz/1e9:.2f} GB", flush=True)


def _nvsmi_used_mb_local_gpu():
    """Sample nvidia-smi for the local rank's GPU.  Returns used-MB int or 0.

    Uses ``CUDA_VISIBLE_DEVICES`` (or falls back to GPU 0) to query just
    this rank's GPU rather than the whole node.  Updates module-level
    ``_NVSMI_PEAK_MB`` running max.  Silently returns 0 on any failure
    (nvidia-smi missing, parse error, timeout) — never raises.
    """
    global _NVSMI_PEAK_MB, _NVSMI_LAST_MB
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            gpu_idx = cvd.split(",")[0].strip()
        else:
            gpu_idx = "0"
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_idx}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        mb = int(out.stdout.strip().split("\n")[0])
        _NVSMI_LAST_MB = mb
        if mb > _NVSMI_PEAK_MB:
            _NVSMI_PEAK_MB = mb
        return mb
    except Exception:
        return 0
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
from .load_wfns import load_centroids_band_chunked
from .wfn_transforms import to_rchunk_inner
from jax.experimental import io_callback as _io_callback


# ============================================================================
# Open-spin pair density: P_k,ab(μ, ν) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(ν)
# ============================================================================
#
# Single rank-5 pair-density path used by every channel (charge γ̃^0 = I_4
# AND transverse γ̃^i = α^i).  The (αβ) spin axes stay OPEN through the
# pair-density and IFFT steps; γ̃·γ̃ contraction happens at the post-IFFT
# reduction inside :func:`c_q_from_psi_sm` / :func:`z_q_from_psi_sm`
# (and at q=0 inside :func:`gram_q0_from_pair`).
#
# For γ̃^0 = γ̃^0 = I_4, ``gamma_double_contract`` short-circuits to the
# Σ_{αβ} P_l_conj·P_r Frobenius reduction (no gather, no phase mul) — see
# :func:`common.gamma_matrices.gamma_double_contract`.
#
# For γ̃^i, γ̃^j = α^l, α^l' it produces
#
#     C_q^{μ_L, ν_L}(μ,ν) = Σ_{k, αβα'β'} P*_{αβ}(μ,ν;k-q)
#                              · γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
#                              · P_{α'β'}(μ,ν;k)
#
# The :func:`pair_density` standalone (rank-5 P_k_ab carrier) is kept
# for ``centroid.pivoted_cholesky``'s q=0 valence-conduction Gram —
# its caller can hold the full P_k_ab in HBM at the centroid-selection
# stage.  The hot CCT/ZCT path in :func:`fit_zeta_to_h5` never
# materialises P_k_ab globally; everything happens inside the
# monolithic shard_map kernels below.

_pair_density_cache = {}
_pair_pipeline_sm_cache = {}


def pair_density(
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Open-spin pair density P_k,ab(μ, col) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(col).

	Spin axes (a,b) are kept open; γ̃ is applied downstream at the C_q
	or Z_q post-IFFT reduction step.

	Inputs:
	    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
	    psi_rcol_Y: (nk, nb, ns, n_col) with P(None, None, None, 'y')

	Output:
	    P_k_ab: (nk, ns, ns, n_rmu, n_col) with P(None, None, None, 'x', 'y')

	einsum: ``'kmna,knbr->kabmr'``.
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('pair_density', id(mesh_xy), nk, n_rmu, nb, ns, n_col)

	if cache_key not in _pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _pair_density(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return jnp.einsum('kmna,knbr->kabmr', psi_L, psi_R, optimize=True)

		_pair_density_cache[cache_key] = _pair_density

	return _pair_density_cache[cache_key](psi_rmuT_X, psi_rcol_Y)


# Cache for ISDF pipeline jitted functions
_isdf_pipeline_cache = {}



# ============================================================================
# q=0 Gram matrix for pivoted-Cholesky centroid selection
# ============================================================================
# Drops the k→q FFT pair from the CCT formula: at q=0 the k-sum IS
# the answer.  Used by :mod:`centroid.pivoted_cholesky` to score
# candidate centroids on valence × conduction pair products.

def gram_q0_from_pair(
	P_v_k: jax.Array,
	P_c_k: jax.Array,
	k_weights: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	mesh_xy: Mesh,
) -> jax.Array:
	"""q=0 valence-conduction pair-product Gram from open-spin pair densities.

	Mathematically (q=0 special case of the CCT-over-k structure):

	    G(μ,ν) = Σ_k w_k · [Σ_{αβα'β'} γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
	                          · P_v_{αβ}(μ,ν;k)*  · P_c_{α'β'}(μ,ν;k)]

	γ̃ identity short-circuit: pass ``gamma_L=None`` (and/or
	``gamma_R=None``) for charge / left-only / right-only sides.
	Both None → Σ_{αβ} P_v* · P_c, the historical pivoted-Cholesky
	candidate Gram in open-spin form.  γ̃^μ is monomial — each non-
	identity contraction is one ``jnp.take`` + element-wise phase
	multiply, not a 4×4 matmul.

	Used by :mod:`centroid.pivoted_cholesky`.

	Args:
		P_v_k: (nk, ns, ns, n_rmu, n_rmu) complex, valence open-spin pair
			density (output of :func:`pair_density` on the valence band
			window), sharded ``P(None, None, None, 'x', 'y')``.
		P_c_k: (nk, ns, ns, n_rmu, n_rmu) complex, conduction window,
			same layout.
		k_weights: (nk,) real, k-point weights (IBZ weights summing to 1,
			or 1/nk_tot for each full-BZ k-point).
		gamma_L, gamma_R: ``(perm, phase)`` tuples or ``None`` (=identity).
		mesh_xy: ('x','y') device mesh, same as the pair densities.

	Returns:
		G: (n_rmu, n_rmu) complex Hermitian PSD, sharded ``P('x','y')``.
	"""
	nk, ns1, ns2, n_rmu, _ = P_v_k.shape
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None
	cache_key = ('gram_q0_from_pair', id(mesh_xy), nk, ns1, ns2, n_rmu,
	             lhs_id, rhs_id)

	if cache_key not in _isdf_pipeline_cache:
		in_spin = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P('x', 'y'))
		kw_rep = NamedSharding(mesh_xy, P())
		rep = NamedSharding(mesh_xy, P())

		_lhs_id = lhs_id
		_rhs_id = rhs_id

		@partial(jax.jit,
		         in_shardings=(in_spin, in_spin, kw_rep, rep, rep, rep, rep),
		         out_shardings=out_xy)
		def _gram_q0(P_v, P_c, kw, perm_L_, phase_L_, perm_R_, phase_R_):
			# γ̃-contracted spin reduction (5 → 3 rank, dropping (a,b)).
			# spin axes are (1, 2) on the (k, a, b, μ, ν) layout.
			prod = gamma_double_contract(
				jnp.conj(P_v), P_c,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(1, 2),
			)
			G = jnp.sum(kw[:, None, None] * prod, axis=0)
			# Symmetrize: q=0 Gram is Hermitian by construction; fp roundoff
			# can break it.  Cheap fix.
			G = 0.5 * (G + jnp.conj(G.T))
			return G

		_isdf_pipeline_cache[cache_key] = _gram_q0

	if lhs_id:
		perm_L = jnp.arange(ns1, dtype=jnp.int32)
		phase_L = jnp.ones(ns1, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns2, dtype=jnp.int32)
		phase_R = jnp.ones(ns2, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	return _isdf_pipeline_cache[cache_key](
		P_v_k, P_c_k, k_weights, perm_L, phase_L, perm_R, phase_R)


# ============================================================================
# 2D Blocked Cholesky Solver - memory efficient for large n_rmu
# ============================================================================

# Cache for 2D Cholesky functions
_chol_2d_cache = {}


# ============================================================================
# Full zeta fitting pipeline with z-chunk loop and HDF5 output
# ============================================================================


# ============================================================================
# Monolithic shard_map pair pipelines — pair density + IFFT + γ̃·γ̃ + FFT
# fused into ONE shard_map.  Take wavefunction tensors directly
# (skipping the standalone rank-5 pair-density buffer); inside the
# shard_map region everything is local-per-rank and the FFTs run via
# direct ``jnp.fft.ifftn`` / ``jnp.fft.fftn`` calls (no nested
# ``make_flat_k_*`` helper — same approach as
# ``wfn_transforms.to_rchunk``).  No nested shard_maps, no helper
# boundary that could let XLA re-globalise the pair density.
#
# Why monolithic: a decomposed chain (pair density → IFFT → γ̃-contract
# → FFT each as its own jit) lands at 5 concurrent rank-5 pair-density
# slots in XLA's BufferAssignment (~4 GiB each at MoS2 3×3 bispinor:
# ``nk · ns² · n_rmu_local · col_local · 16``), pegging the kernel
# preallocated-temp peak at 21 GiB on a 28 GiB budget.  Folding into
# one shard_map drops the slot count to 3 (saving the standalone IFFT
# outputs + the gamma-contract intermediate), bringing the kernel peak
# to 13 GiB.
# ============================================================================


def c_q_from_psi_sm(
	psi_l_X: jax.Array,
	psi_l_Y: jax.Array,
	psi_r_X: jax.Array,
	psi_r_Y: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""C_q built from ψ directly inside one monolithic shard_map.

	Inputs:
	    psi_l_X, psi_r_X : (nk, n_rmu, nb, ns) sharded ``P(None, 'x', None, None)``
	    psi_l_Y, psi_r_Y : (nk, nb, ns, n_col) sharded ``P(None, None, None, 'y')``
	    gamma_L, gamma_R : ``(perm, phase)`` tuples or ``None`` (= γ̃^0 = I).
	    kgrid            : (nkx, nky, nkz).
	Output:
	    C_q              : (nq, n_rmu, n_col) sharded ``P(None, 'x', 'y')``.

	``n_col == n_rmu`` for CCT (square centroid).
	"""
	nkx, nky, nkz = kgrid
	nk = int(psi_l_X.shape[0])
	n_rmu = int(psi_l_X.shape[1])
	nb_l = int(psi_l_X.shape[2])
	nb_r = int(psi_r_X.shape[2])
	ns = int(psi_l_X.shape[3])
	n_col = int(psi_l_Y.shape[3])
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = ('c_q_from_psi_sm', id(mesh_xy), nk, n_rmu, n_col, ns,
	             nb_l, nb_r, nkx, nky, nkz, lhs_id, rhs_id)
	if cache_key not in _pair_pipeline_sm_cache:
		_lhs_id = lhs_id
		_rhs_id = rhs_id
		L_spec = P(None, 'x', None, None)
		R_spec = P(None, None, None, 'y')
		out_spec = P(None, 'x', 'y')

		@partial(shard_map, mesh=mesh_xy,
		         in_specs=(L_spec, R_spec, L_spec, R_spec,
		                   P(), P(), P(), P()),
		         out_specs=out_spec,
		         check_rep=False)
		def _local(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_,
		           perm_L_, phase_L_, perm_R_, phase_R_):
			# Per-rank shapes (n_rmu_local on 'x', n_col_local on 'y',
			# bands replicated):
			#   psi_l_X_ : (nk, n_rmu_local, nb_l, ns)
			#   psi_l_Y_ : (nk, nb_l, ns, n_col_local)
			#
			# Einsum output order matters for memory: the gemm naturally
			# produces ``(k, ns_l · col_local, ns_r · μ_local)`` with the
			# bitcast factoring as ``(k, ns_l, col_local, μ_local, ns_r)``.
			# We pick output spec ``'karmb'`` (k, ns_l, col, μ, ns_r) so
			# the rank-7 reshape that feeds the IFFT is a pure bitcast —
			# no extra 4-GiB buffer for the rank-3 → rank-7 transpose
			# (which the natural ``'kabmr'`` order forced).  Verified
			# from HLO: dropped one of the three rank-5 lifetime slots.
			mu_loc = psi_l_X_.shape[1]
			col_loc = psi_l_Y_.shape[3]
			P_l = jnp.einsum(
				'kmna,knbr->karmb', psi_l_X_, psi_l_Y_, optimize=True)
			P_r = jnp.einsum(
				'kmna,knbr->karmb', psi_r_X_, psi_r_Y_, optimize=True)
			# Split k → (kx, ky, kz) — bitcast given the above layout.
			# Rank-7: (kx, ky, kz, ns_l, col, μ, ns_r).
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns, col_loc, mu_loc, ns)
			del P_l
			P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
			P_l_R_conj = jnp.conj(P_l_R)
			del P_l_3d, P_l_R
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns, col_loc, mu_loc, ns)
			del P_r
			P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')
			del P_r_3d
			# Reduce over the spin axes (3=ns_l, 6=ns_r) of the rank-7
			# form.  Output rank-5: (kx, ky, kz, col, μ).
			C_R = gamma_double_contract(
				P_l_R_conj, P_r_R,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(3, 6),
			)
			del P_l_R_conj, P_r_R
			C_q_3d = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='forward')
			# Reshape back to (nk, col, μ); transpose final two axes
			# to satisfy out_spec ``P(None, 'x', 'y')`` for (nk, μ, col).
			# This transpose acts on the rank-3 reduced form (~16 MB),
			# not the rank-5 pair density.
			return jnp.transpose(
				C_q_3d.reshape(nkx * nky * nkz, col_loc, mu_loc),
				(0, 2, 1))

		@jax.jit
		def fn(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_, pL, phL, pR, phR):
			return _local(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_,
			              pL, phL, pR, phR)

		_pair_pipeline_sm_cache[cache_key] = fn

	if lhs_id:
		perm_L = jnp.arange(ns, dtype=jnp.int32)
		phase_L = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns, dtype=jnp.int32)
		phase_R = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	return _pair_pipeline_sm_cache[cache_key](
		psi_l_X, psi_l_Y, psi_r_X, psi_r_Y,
		perm_L, phase_L, perm_R, phase_R)


def z_q_from_psi_sm(
	psi_l_X: jax.Array,
	psi_r_X: jax.Array,
	psi_G_store,
	*,
	band_chunk_ranges: tuple[tuple[int, int], ...],
	band_range_left: tuple[int, int],
	band_range_right: tuple[int, int],
	r_start_dyn,
	r_chunk_size: int,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""Z_q built from ψ via a streaming-scan pair density inside one shard_map.

	Round 6 redesign (`round5_unified_plan.md` §2.10 / §6.5).  Replaces
	the all-at-once einsum that consumed pre-computed ``psi_l_Y`` /
	``psi_r_Y`` with a ``lax.scan`` over band-chunks inside the
	``shard_map`` body.  Per iter:

	  1. ``io_callback`` pulls this rank's 1/P bands of bc ``i`` from
	     :class:`PsiGStore`'s host tile (band-flat-sharded over the
	     full ``('x','y')`` mesh).
	  2. :func:`common.wfn_transforms.to_rchunk_inner` does the local
	     IFFT + per-rank r-slab (``r0_local = r_start + axis_index('y')
	     * r_loc``).
	  3. ``lax.all_gather(axis_name=('x','y'), axis=1, tiled=True)``
	     aligns the band axis with ``psi_l_X`` / ``psi_r_X``'s
	     band-replicated layout.  IFFT-FIRST; gather-first would blow
	     the FFT box to ~80 GB / rank.
	  4. L/R per-bc band masks (mask approach — ``jnp.where`` on a
	     rank-local axis).
	  5. Two einsums into rank-5 carries ``(P_l_acc, P_r_acc)``.

	Post-scan: existing IFFT(k) → γ̃·γ̃ → FFT(k) tail (byte-identical
	to the pre-rewrite body — only the front of the body changes per
	plan §2.7).

	Inputs:
	    psi_l_X, psi_r_X    : ``(nk, n_rmu, nb_l, ns)`` /
	                          ``(nk, n_rmu, nb_r, ns)`` sharded
	                          ``P(None, 'x', None, None)`` (μ on 'x',
	                          bands replicated).
	    psi_G_store         : :class:`PsiGStore` — closure-captured;
	                          provides ``_slice_local_tile_bc`` /
	                          ``g_index`` / ``kvecs_frac``.  NOT a jit
	                          argument.
	    band_chunk_ranges   : tuple of (b_lo, b_hi) global band indices.
	    band_range_left     : (L_lo, L_hi) global L-window.  Must
	                          satisfy nb_l == L_hi - L_lo.
	    band_range_right    : (R_lo, R_hi) global R-window.
	    fft_grid            : (nx, ny, nz).
	    r_start_dyn         : int32 scalar — flat-r start of the chunk.
	    r_chunk_size        : static int — full r-chunk extent.  Per
	                          rank slab is ``r_chunk_size // p_y``.
	    gamma_L, gamma_R    : ``(perm, phase)`` tuples or ``None``
	                          (= γ̃^0 = I).
	    kgrid               : (nkx, nky, nkz).
	Output:
	    Z_q                 : (nq, n_rmu, n_zchunk) sharded
	                          ``P(None, 'x', 'y')``.
	"""
	from .psi_G_store import _PSI_G_FLAT_SPEC  # noqa: F401  (sharding contract)

	fft_grid = tuple(int(s) for s in psi_G_store.meta.fft_grid)
	nkx, nky, nkz = kgrid
	nk = int(psi_l_X.shape[0])
	n_rmu = int(psi_l_X.shape[1])
	nb_l = int(psi_l_X.shape[2])
	nb_r = int(psi_r_X.shape[2])
	ns = int(psi_l_X.shape[3])
	n_zchunk = int(r_chunk_size)
	p_x = int(mesh_xy.shape['x'])
	p_y = int(mesh_xy.shape['y'])
	if n_zchunk % p_y != 0:
		raise ValueError(
			f"z_q_from_psi_sm: r_chunk_size={n_zchunk} not divisible by "
			f"p_y={p_y} (out_spec=P(None,'x','y') requires this).")
	r_loc = n_zchunk // p_y

	bcr = tuple((int(lo), int(hi)) for (lo, hi) in band_chunk_ranges)
	L_lo_g = int(band_range_left[0]); L_hi_g = int(band_range_left[1])
	R_lo_g = int(band_range_right[0]); R_hi_g = int(band_range_right[1])
	if L_hi_g - L_lo_g != nb_l:
		raise ValueError(
			f"z_q_from_psi_sm: psi_l_X.shape[2]={nb_l} != L_hi - L_lo "
			f"= {L_hi_g - L_lo_g}.")
	if R_hi_g - R_lo_g != nb_r:
		raise ValueError(
			f"z_q_from_psi_sm: psi_r_X.shape[2]={nb_r} != R_hi - R_lo "
			f"= {R_hi_g - R_lo_g}.")

	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = (
		'z_q_from_psi_sm_streaming', id(mesh_xy), id(psi_G_store),
		nk, n_rmu, n_zchunk, ns, nb_l, nb_r, nkx, nky, nkz,
		lhs_id, rhs_id, bcr, (L_lo_g, L_hi_g), (R_lo_g, R_hi_g),
		tuple(int(s) for s in fft_grid),
	)
	if cache_key not in _pair_pipeline_sm_cache:
		_lhs_id = lhs_id
		_rhs_id = rhs_id
		_psi_G_store = psi_G_store
		fft_grid_t = tuple(int(s) for s in fft_grid)

		# Static per-bc tables: global band-axis offsets for the
		# bc-aligned scan body.  Each iter gathers ``bpd_max_global =
		# P · _bpd_max`` bands (the bc's full band width), so the L/R
		# masks index into a static-size axis.
		P_total = p_x * p_y
		bpd_max = int(_psi_G_store._bpd_max)
		bpd_max_global = bpd_max * P_total          # padded global bc band count
		n_bc = len(bcr)
		# Per-bc tables.  Built as np arrays here (NOT jnp.asarray) so
		# they enter the shard_map body via numpy → jnp lift inside the
		# Manual-mode body.  Closure-captured Auto-sharded jax.Arrays
		# inside a Manual-mode shard_map body trigger a mesh-context
		# mismatch (same issue as kvecs_frac in the earlier debug);
		# wrapping them as numpy constants and lifting inside the body
		# treats them as concrete constants from the body's perspective.
		_b_lo_global_np = np.asarray(
			[lo for (lo, _hi) in bcr], dtype=np.int32)
		_b_hi_global_np = np.asarray(
			[hi for (_lo, hi) in bcr], dtype=np.int32)
		# psi_l_X / psi_r_X are sized to their L/R window; per-bc slice
		# offset within them is `bc.lo - L_lo_g` / `bc.lo - R_lo_g`,
		# which can be NEGATIVE when bc starts below the window AND
		# the slice (offset, offset+bpd_max_global) can extend PAST
		# the window when bc spans the upper boundary or the final bc
		# is short.  We pre-pad psi_l_X / psi_r_X at BOTH ends with
		# zero rows so that the dynamic_slice never goes out of
		# bounds — which would otherwise trigger XLA's silent clamp
		# (start clamped to ``axis_size - slice_size``), producing
		# *physically wrong bands* that the L/R mask CANNOT recover
		# (the mask's index assumes the slice covers
		# ``[bc.lo, bc.lo+bpd_max_global)`` but the clamp returns
		# something else).  See round6_discussion.md:506 BLOCKER from
		# Agent 4.  Pad rows correspond to out-of-window global bands;
		# the L/R mask zeros their contribution to the einsum
		# (math-neutral) — same contract as the front-pad rows.
		front_pad_l = max(
			(max(0, L_lo_g - lo) for (lo, _hi) in bcr), default=0)
		front_pad_r = max(
			(max(0, R_lo_g - lo) for (lo, _hi) in bcr), default=0)
		# After front-pad, offset = bc.lo - L_lo_g + front_pad_l (always ≥ 0).
		_psi_l_X_bc_offset_np = np.asarray(
			[lo - L_lo_g + front_pad_l for (lo, _hi) in bcr],
			dtype=np.int32)
		_psi_r_X_bc_offset_np = np.asarray(
			[lo - R_lo_g + front_pad_r for (lo, _hi) in bcr],
			dtype=np.int32)
		# Back-pad: the slice (offset, offset+bpd_max_global) must fit
		# entirely within the padded array (front_pad + nb + back_pad).
		# The largest end-offset across all bcs determines the back-pad.
		_max_end_l = int(max(
			(off + bpd_max_global for off in _psi_l_X_bc_offset_np),
			default=0))
		_max_end_r = int(max(
			(off + bpd_max_global for off in _psi_r_X_bc_offset_np),
			default=0))
		back_pad_l = max(0, _max_end_l - (front_pad_l + nb_l))
		back_pad_r = max(0, _max_end_r - (front_pad_r + nb_r))
		ngkmax = int(_psi_G_store._per_rank_shape[3])

		# Slicer needs static `out_sds`; close over the padded shape.
		_per_rank_bc_shape = (nk, bpd_max, ns, ngkmax)
		_slicer_out_sds = jax.ShapeDtypeStruct(
			_per_rank_bc_shape, jnp.complex128)

		def _slicer_host(x_idx, y_idx, bc_idx):
			return _psi_G_store._slice_local_tile_bc(x_idx, y_idx, bc_idx)

		L_spec = P(None, 'x', None, None)
		out_spec = P(None, 'x', 'y')
		# g_index, kvecs_frac: replicated.  Pass through shard_map's
		# in_specs (NOT closure) so JAX sees Manual-mode-compatible
		# access inside the body (closure-captured Auto-sharded arrays
		# trip a mesh-context mismatch under multi-device shard_map).
		g_index_spec    = P(None, None, None, None)
		kvecs_frac_spec = P(None, None)

		@partial(shard_map, mesh=mesh_xy,
		         in_specs=(L_spec, L_spec, P(), P(), P(), P(), P(),
		                   g_index_spec, kvecs_frac_spec),
		         out_specs=out_spec,
		         check_rep=False)
		def _local(psi_l_X_, psi_r_X_, perm_L_, phase_L_,
		           perm_R_, phase_R_, r_start_,
		           g_index_dev, kvecs_frac_dev):
			# Per-rank shapes:
			#   psi_l_X_ : (nk, n_rmu_loc, nb_l, ns)   μ on 'x' (replicated bands)
			#   psi_r_X_ : (nk, n_rmu_loc, nb_r, ns)
			# Carry (rank-local, NO sharding annotation):
			#   P_l_acc  : (nk, ns, r_loc, mu_loc, ns)
			# r_loc = n_zchunk / p_y per §2.3 (out_spec='y' on n_zchunk).
			x_idx = jax.lax.axis_index('x')
			y_idx = jax.lax.axis_index('y')
			mu_loc = psi_l_X_.shape[1]

			# Lift per-bc static tables to jnp.array INSIDE the
			# Manual-mode body so they're treated as Manual-mode
			# concrete constants (vs Auto-sharded closure jax.Arrays
			# which trip a mesh-context mismatch).
			b_lo_global_arr = jnp.asarray(_b_lo_global_np)
			b_hi_global_arr = jnp.asarray(_b_hi_global_np)
			psi_l_X_bc_offset = jnp.asarray(_psi_l_X_bc_offset_np)
			psi_r_X_bc_offset = jnp.asarray(_psi_r_X_bc_offset_np)

			# Pre-pad psi_l_X_ / psi_r_X_ at the front with front_pad_*
			# zero bands so the per-bc offset is always non-negative
			# (handles the bc.lo < L_lo_g / R_lo_g case where the
			# X-side offset would otherwise be negative; bands at
			# position [0, front_pad) are zeros, masked-zero in the
			# einsum).  Static front_pad_* baked at trace.
			# Pad both ends.  Front-pad covers bc.lo < window_lo
			# (negative offset case); back-pad covers
			# bc.lo + bpd_max_global > window_lo + nb (out-of-bounds
			# end case — XLA's dynamic_slice would otherwise silently
			# clamp the start, producing wrong bands the L/R mask
			# can't recover; see round6_discussion.md:506 BLOCKER).
			if front_pad_l > 0 or back_pad_l > 0:
				psi_l_X_padded = jnp.pad(
					psi_l_X_,
					((0, 0), (0, 0), (front_pad_l, back_pad_l), (0, 0)))
			else:
				psi_l_X_padded = psi_l_X_
			if front_pad_r > 0 or back_pad_r > 0:
				psi_r_X_padded = jnp.pad(
					psi_r_X_,
					((0, 0), (0, 0), (front_pad_r, back_pad_r), (0, 0)))
			else:
				psi_r_X_padded = psi_r_X_

			# r-slab strategy: each y-rank ultimately owns ``r_loc``
			# positions of the r-chunk (out_spec=P(None, 'x', 'y') on
			# n_zchunk).  We CANNOT slice the r-axis per-rank BEFORE
			# the all_gather over bands — that would mix r-slabs from
			# different y-ranks at the same gathered band position
			# (r-incoherence at the einsum).  Instead: per rank
			# compute the FULL r-chunk in psi_Y_local, gather bands,
			# THEN slice the r-axis to this y-rank's per-rank slab.
			# Per-rank cost of full-r psi_Y_local is bigger by p_y vs
			# the r_loc version, but XLA's scan-internal allocator
			# aliases the per-iter slab across iters → single slot.
			r0_y_offset = y_idx * jnp.int32(r_loc)

			P_l_init = jnp.zeros(
				(nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)
			P_r_init = jnp.zeros(
				(nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)

			def body(carry, bc_idx):
				P_l_acc, P_r_acc = carry
				# (1) io_callback: this rank's 1/P bands of bc bc_idx.
				#     ordered=False per §3.4 (lax.scan(unroll=1) gives
				#     sequential per-rank execution at runtime; ordered
				#     is a perf knob, not correctness).
				psi_G_bc_local = _io_callback(
					_slicer_host, _slicer_out_sds,
					x_idx, y_idx, bc_idx,
					ordered=False)
				# (2) IFFT + FULL r-chunk slab (NOT per-rank r_loc — see
				#     "r-slab strategy" comment above).  Local FFT box
				#     per rank is c128[nk, bpd_max, ns, n_rtot] ·
				#     cuFFT_scratch; the full-r slab is c128[nk,
				#     bpd_max, ns, n_zchunk] per rank.  Both per-iter,
				#     aliased across iters by XLA's scan-internal
				#     allocator → single slot of each.
				psi_Y_bc_local_full_r = to_rchunk_inner(
					psi_G_bc_local, g_index_dev, fft_grid_t,
					r_start_, n_zchunk,
					kvecs_frac=kvecs_frac_dev, norm="ortho")
				# (3) all_gather across both mesh axes along the band
				#     axis.  IFFT-FIRST per §2.9 (gather-first → 80 GB
				#     FFT box, infeasible).  Output: (nk, P·bpd_max,
				#     ns, n_zchunk) on every rank.
				psi_Y_bc_full_r = jax.lax.all_gather(
					psi_Y_bc_local_full_r, axis_name=('x', 'y'),
					axis=1, tiled=True)
				# (3b) Slice the r-axis to THIS y-rank's r_loc slab.
				#      MUST happen AFTER gather so the band axis +
				#      r axis are coherent (each gathered band's r
				#      values come from the SAME source rank's full
				#      r-chunk computation).
				psi_Y_bc = jax.lax.dynamic_slice_in_dim(
					psi_Y_bc_full_r, r0_y_offset, r_loc, axis=3)
				# (4) L/R global-index masks (mask approach per §2.5).
				#     bc_valid handles short final bc (pad rows = zero).
				g_axis = (b_lo_global_arr[bc_idx]
				          + jnp.arange(bpd_max_global, dtype=jnp.int32))
				bc_valid = g_axis < b_hi_global_arr[bc_idx]
				l_mask = ((g_axis >= L_lo_g) & (g_axis < L_hi_g) & bc_valid)
				r_mask = ((g_axis >= R_lo_g) & (g_axis < R_hi_g) & bc_valid)
				psi_l_Y_bc = jnp.where(
					l_mask[None, :, None, None], psi_Y_bc, 0)
				psi_r_Y_bc = jnp.where(
					r_mask[None, :, None, None], psi_Y_bc, 0)
				# (5) psi_l_X / psi_r_X per-bc slice (band axis
				#     replicated → purely local).  Static slice length
				#     bpd_max_global so XLA can fold; offset traced.
				#     Slice from the FRONT-PADDED psi_*_X_padded so the
				#     offset is always non-negative even for bcs
				#     starting below the L/R window start.
				psi_l_X_bc = jax.lax.dynamic_slice_in_dim(
					psi_l_X_padded, psi_l_X_bc_offset[bc_idx],
					bpd_max_global, axis=2)
				psi_l_X_bc = jnp.where(
					l_mask[None, None, :, None], psi_l_X_bc, 0)
				psi_r_X_bc = jax.lax.dynamic_slice_in_dim(
					psi_r_X_padded, psi_r_X_bc_offset[bc_idx],
					bpd_max_global, axis=2)
				psi_r_X_bc = jnp.where(
					r_mask[None, None, :, None], psi_r_X_bc, 0)
				# (6) Two einsums into the carries (interleaved single
				#     scan per §2.10; with the small carry the γ̃
				#     contract dominates peak — serialization bought
				#     nothing).
				delta_P_l = jnp.einsum(
					'kmna,knbr->karmb',
					psi_l_X_bc, psi_l_Y_bc, optimize=True)
				delta_P_r = jnp.einsum(
					'kmna,knbr->karmb',
					psi_r_X_bc, psi_r_Y_bc, optimize=True)
				return (P_l_acc + delta_P_l, P_r_acc + delta_P_r), None

			# DO NOT unroll — the FFT-box and psi_G_bc aliasing depends
			# on per-iter sequential lifetime.  unroll=1 keeps the
			# WhileOp atomic and lets XLA's scan-internal allocator
			# reuse the slot (§3.5).
			(P_l, P_r), _ = jax.lax.scan(
				body, (P_l_init, P_r_init),
				jnp.arange(n_bc, dtype=jnp.int32))

			# Post-pair pipeline (byte-identical to today's tail per §2.7).
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
			del P_l
			P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
			P_l_R_conj = jnp.conj(P_l_R)
			del P_l_3d, P_l_R
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
			del P_r
			P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')
			del P_r_3d
			Z_R = gamma_double_contract(
				P_l_R_conj, P_r_R,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(3, 6),
			)
			del P_l_R_conj, P_r_R
			Z_q_3d = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='forward')
			return jnp.transpose(
				Z_q_3d.reshape(nkx * nky * nkz, r_loc, mu_loc),
				(0, 2, 1))

		@jax.jit
		def fn(psi_l_X_, psi_r_X_, pL, phL, pR, phR, r_start_,
		        g_index_, kvecs_frac_):
			return _local(psi_l_X_, psi_r_X_, pL, phL, pR, phR, r_start_,
			              g_index_, kvecs_frac_)

		_pair_pipeline_sm_cache[cache_key] = fn

	if lhs_id:
		perm_L = jnp.arange(ns, dtype=jnp.int32)
		phase_L = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns, dtype=jnp.int32)
		phase_R = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	r_start_arg = (jnp.int32(int(r_start_dyn))
	                if isinstance(r_start_dyn, (int, np.integer))
	                else r_start_dyn)
	return _pair_pipeline_sm_cache[cache_key](
		psi_l_X, psi_r_X, perm_L, phase_L, perm_R, phase_R, r_start_arg,
		psi_G_store.g_index, psi_G_store.kvecs_frac)


# Backward-compat shim removed — old z_q_from_psi_sm signature
# (consumed pre-computed psi_l_Y / psi_r_Y) is gone.  Call sites
# updated in `_make_fit_one_rchunk_kernel._kernel`.


def _identity_pad_block_diagonal(
    M: jax.Array,
    *,
    n_rmu_logical: int,
    mesh_xy: Mesh,
) -> jax.Array:
    """Add identity to the pad-block diagonal of a square N_μ² matrix.

    ``M`` has shape ``(nq, n_rmu, n_rmu)`` at PADDED μ extent with
    zero pad rows/cols (the Phase 3a contract: bilinear in zero-padded
    ψ ⇒ M's pad rows/cols are exact zeros).  This helper adds 1 to the
    diagonal entries in positions ``[n_rmu_logical, n_rmu)``, leaving
    the logical block exactly intact.  Result: ``M_id_pad =
    block_diag(M_log, I_pad)`` — block-diagonal with the input's
    logical block on top-left and identity on bottom-right.

    Why this matters:  Cholesky and LU on the identity-padded matrix
    produce factorisations whose **logical block is bit-identical** to
    the factorisation of the un-padded logical-only matrix.  The
    standard recursion for ``L[i, j]`` (Cholesky) and the column-
    pivoting LU never read across the zero off-diagonal pad blocks,
    and ``√1 = 1`` exactly in IEEE 754 so the pad-block factor is
    exactly identity.  The downstream back-solve sees ``Z`` with zero
    pad rows (bilinear in zero-padded ψ ⇒ zero pad rows), and
    ``[L_log 0; 0 I][y_log; y_pad] = [Z_log; 0]`` gives
    ``y_pad = 0`` and ``y_log = L_log⁻¹ Z_log`` — logical solve
    unchanged.

    This is NOT ridge regularisation on C_q (which would corrupt the
    logical block).  The identity is added ONLY to the pad-block
    diagonal; the logical block is untouched.

    Output sharding is ``P(None, 'x', 'y')`` (n_rmu_padded is
    mesh-divisible by construction so single-axis sharding on each
    μ-dim works at any padded extent).  When ``n_rmu_logical ==
    n_rmu`` (no pad), the function is a no-op pass-through with the
    sharding constraint reapplied.
    """
    nq, n_rmu, n_rmu2 = M.shape
    if n_rmu != n_rmu2:
        raise ValueError(f"_identity_pad_block_diagonal expects square M; got {M.shape}")
    if n_rmu_logical > n_rmu:
        raise ValueError(
            f"n_rmu_logical={n_rmu_logical} > input extent {n_rmu}")
    sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    if n_rmu_logical == n_rmu:
        return jax.lax.with_sharding_constraint(M, sharding)
    # Build a (n_rmu, n_rmu) diagonal matrix whose only non-zeros sit
    # on positions ``[n_rmu_logical, n_rmu)`` of the diagonal.
    idx = jnp.arange(n_rmu)
    pad_diag_mask = (idx >= n_rmu_logical).astype(M.dtype)
    eye_pad = jnp.diag(pad_diag_mask)
    M_id_pad = M + eye_pad[None, :, :]
    return jax.lax.with_sharding_constraint(M_id_pad, sharding)


def _resolve_solver_kind_charge(mesh_xy: Mesh, override: str = "auto") -> str:
    """Pick the charge-channel ζ-fit solver: cuSolverMp distributed
    potrf+potrs vs the in-tree shard_map 2D-blocked Cholesky + per-q
    triangular solve.

    Default policy (2026-05-12): use cuSolverMp on **true 2D meshes**
    (px≥2 AND py≥2); the in-tree ``sharded_cholesky`` runs many small
    NCCL all-reduces per panel which become the dominant GPU stream
    consumer at production scale (CrI3 6×6 80 Ry n_rmu≈1500 sees ~tens
    of seconds in the panel loop).  cuSolverMp bundles the whole
    distributed Cholesky into one FFI call per q.

    Tradeoff: at small scales (MoS2 3×3, n_rmu=640, 2×2 mesh) the FFI
    setup overhead exceeds the savings — measured cholesky 3.6 s
    (cuSolverMp) vs 1.3 s (sharded) on a 2×2 mesh.  Total wall
    difference ~0.5 s, within run-to-run noise.  We accept the small
    overhead for the larger-scale win.

    Override via cohsex.in ``cusolvermp_charge``:
      ``off`` → force sharded.
      ``on``  → force cuSolverMp (still falls back on 1D meshes).
      ``auto`` (default) → cuSolverMp on true 2D, sharded otherwise.
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    if override == 'off':
        return 'sharded_cholesky'
    if override == 'on':
        return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'

    # auto (or unrecognised) → default policy.
    return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'


def _resolve_solver_kind_transverse(mesh_xy: Mesh, override: str = "auto") -> str:
    """Pick the transverse-channel ζ-fit solver: cuSolverMp distributed
    getrf+getrs vs the in-tree per-q ``jnp.linalg.solve`` + ridge.

    Default policy (2026-05-12): mirrors the charge-channel resolver —
    use cuSolverMp on **true 2D meshes** (px≥2 AND py≥2).  cuSolverMp
    0.7.2 fixes the earlier 2D-grid getrf/getrs correctness bug
    (validated end-to-end on MoS2 3×3 bispinor at 2×2 mesh; see
    ``src/ffi/cusolvermp/cpp/batched_solve_lu_ffi.cc`` for history).

    Tradeoff: small FFI setup overhead at MoS2 scale (n_rmu=656,
    2×2 mesh).  At CrI3 6×6 80 Ry (n_rmu≈1800, 4×4 mesh) the cuSolverMp
    path is the right tool.

    Override via cohsex.in ``cusolvermp_lu``:
      ``off`` → force per-q ``jnp.linalg.solve``.
      ``on``  → force cuSolverMp (still falls back on 1D meshes).
      ``auto`` (default) → cuSolverMp on true 2D, legacy otherwise.
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    if override == 'off':
        return 'lu'
    if override == 'on':
        return 'cusolvermp_lu' if is_2d else 'lu'

    return 'cusolvermp_lu' if is_2d else 'lu'


def _resolve_solver_kind(
    mesh_xy: Mesh, vertex_mu_L: int, solver_kind: str,
    cusolvermp_charge: str = "auto",
    cusolvermp_lu: str = "auto",
) -> str:
    """Single source of truth for the ``auto`` resolution.  Transverse
    channels (γ̃^i, μ_L≠0) take ``_resolve_solver_kind_transverse``;
    charge channel takes ``_resolve_solver_kind_charge``.
    """
    if solver_kind != 'auto':
        return solver_kind
    if int(vertex_mu_L) != 0:
        return _resolve_solver_kind_transverse(mesh_xy, cusolvermp_lu)
    return _resolve_solver_kind_charge(mesh_xy, cusolvermp_charge)


def factor_c_q(
    C_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
    vertex_mu_L: int = 0,
    n_rmu_logical: int | None = None,
    solver_kind: str = 'auto',
) -> jax.Array:
    """
    Compute system-matrix L_q from CCT matrix.

    For ``vertex_mu_L == 0`` (standard spin-traced path) the CCT is
    Hermitian positive-definite (modulo numerical noise); we run the
    optimized 2D blocked Cholesky and return the lower-triangular
    factor.  Downstream :func:`solve_zeta` then does two
    triangular solves per-q.

    For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
    the CCT is Hermitian but **indefinite** — Cholesky NaNs and the LU
    fallback in :func:`solve_zeta` is required.  In this case
    we skip the factorization here and pass C_q through unchanged; the
    solve routine consumes it via ``jnp.linalg.solve`` on a per-q-batch
    basis (one LU per call, small enough that explicit
    ``lu_factor`` + ``lu_solve`` reuse buys nothing).

    Padded-input path (``n_rmu_logical < C_q.shape[-1]``):
    n_rmu may be padded to mesh divisibility at the boundary so the
    ``P(None, 'x', 'y')`` input sharding is admissible at any logical
    centroid count (e.g. n_rmu_logical = 661 prime → padded to 672 on
    a 4×4 mesh).  By the Phase 3a contract the trailing pad rows/cols
    of C_q are exact zeros (bilinear in zero-padded ψ).  We add
    identity ONLY to the pad-block diagonal in-place — turning C_q
    into a block-diagonal ``[C_log 0; 0 I_pad]`` matrix — and then
    run the same sharded Cholesky / LU path the divisible case uses.
    Cholesky / LU of an identity-padded matrix produces a factor
    whose logical block is bit-identical to the factor of the
    logical-only matrix (the recursion never reads across zero
    off-diagonal pad blocks, and ``√1 = 1`` exactly).  The pad-block
    factor is exactly identity; the back-solve's pad rows of ζ come
    out as zero (because Z's pad rows are zero by the same bilinear
    argument); logical block of ζ is unchanged.

    This is NOT ridge regularisation of C_q.  The logical block is
    untouched; identity is added ONLY to the pad-block diagonal.

    Output sharding is ``P(None, 'x', 'y')`` natively at the padded
    extent — no replication, no slice + embed gymnastics, the chol
    stays sharded across the mesh.

    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix at PADDED μ extent, sharded
            ``P(None, 'x', 'y')``.  ``n_rmu == n_rmu_padded`` (== ∏ p_a
            of the device mesh) so the existing 2D-blocked path
            applies.
        mesh_xy: 2D device mesh.
        block_size: Tile block size (auto if None).
        vertex_mu_L: Lorentz vertex index (0 = spin-traced PSD path,
            1/2/3 = transverse indefinite path).
        n_rmu_logical: Logical centroid count.  When given and
            strictly less than ``C_q.shape[-1]``, the pad-block
            diagonal is set to identity before factorisation.
            ``None`` (default) skips the identity-pad: input == output
            extent and the matrix is assumed to be PSD on its full
            extent (legacy mesh-divisible path).

    Returns:
        L_q: ``(nq, n_rmu, n_rmu)`` at PADDED extent, sharded
        ``P(None, 'x', 'y')``.  Cholesky factor for
        ``vertex_mu_L == 0`` (block-diagonal ``[L_log 0; 0 I_pad]``
        when n_rmu_logical < n_rmu); passthrough identity-padded CCT
        for ``vertex_mu_L ≠ 0``.
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    if n_rmu_logical is None:
        n_rmu_logical = n_rmu
    if n_rmu_logical > n_rmu:
        raise ValueError(
            f"n_rmu_logical={n_rmu_logical} exceeds input extent {n_rmu}")

    # Pad-block identity in-place.  No-op when n_rmu_logical == n_rmu.
    # See ``_identity_pad_block_diagonal`` for why the logical block of
    # the resulting factorisation is bit-identical to a logical-only
    # factorisation.
    C_q = _identity_pad_block_diagonal(
        C_q, n_rmu_logical=n_rmu_logical, mesh_xy=mesh_xy)

    # Indefinite-CCT path: skip the Cholesky outright.  ``solve_zeta``
    # consumes C_q directly via the SVD pseudoinverse / pivoted-LU
    # branch.  After identity-pad the matrix is non-singular at the
    # padded extent; the indefinite logical block is untouched.
    if int(vertex_mu_L) != 0:
        return C_q

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L=0, solver_kind=solver_kind)

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrf on C_q at P(None,'x','y'); returns the raw
        # lower-triangular factor.  Downstream solve_zeta rebuilds the
        # CusolverMpBatchedLowerL handle and dispatches to potrs.
        from ffi.cusolvermp import batched_distributed_cholesky
        L_handle = batched_distributed_cholesky(C_q, mesh=mesh_xy)
        return L_handle.raw

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

    # 2D-blocked path: requires n_rmu divisible into mesh-friendly tiles.
    # The caller is expected to pass C_q at PADDED μ extent
    # (n_rmu_padded ≡ ∏ p_a is mesh-product divisible), in which case
    # this always succeeds.  ``n_rmu_logical`` only adjusts the
    # identity-pad above; it doesn't affect this code path.
    try:
        if block_size is None:
            block_size, J = compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)
        else:
            J = n_rmu // block_size
    except ValueError as exc:
        raise ValueError(
            f"factor_c_q: n_rmu={n_rmu} is not 2D-blocked-Cholesky "
            f"compatible with mesh {Pr}×{Pc} ({exc}). Pass C_q at "
            f"PADDED μ extent (round up to ∏ p_a = world_size) so "
            f"the 2D-blocked path applies."
        ) from exc

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


def _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta: jax.Array, mesh_xy: Mesh) -> jax.Array:
    """Reshard (q_, μ_X, r_Y) → (q_, μ_XY, r_) for the cuSolverMp branches.

    Single mesh axis ``'y'`` moves from the r-axis to the μ-axis (where
    it joins ``'x'`` to form a flat tuple).  All other shardings stay.

    Downstream consumer ``accumulate_rchunk_to_gflat`` wants ζ
    μ-flat-sharded so the FFT box and gflat-accumulator both live at
    ``P(None, ('x','y'), None)``; landing ζ in that layout here means
    the FFT runs sharding-preserving (no further reshard, no
    replicated FFT box).

    Note on overhead: tried both ``@jax.jit(donate_argnums=(0,))``
    closure-wrapping (matching the ``_reshard_z`` pattern above) and a
    module-level decorator with ``static_argnums``.  Neither flipped
    XLA's ``is_sync`` flag on the emitted all-to-all from ``true`` to
    ``false``, and runtime cost was the same either way (~3 ms/call in
    the trace).  The bare ``with_sharding_constraint`` is the simplest
    form for the same emitted HLO; trace shows the reshard is not on
    the critical path at MoS2 3×3 scale.
    """
    return jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, ('x', 'y'), None)))


def _reshard_zeta_r_XY_to_mu_XY(zeta: jax.Array, mesh_xy: Mesh) -> jax.Array:
    """Reshard (q_, μ_, r_XY) → (q_, μ_XY, r_) for the shard_map branch.

    The shard_map triangular-solve naturally lands ζ at
    ``P(None, None, ('x','y'))`` because the solve is parallelised over
    r-columns.  The downstream FFT wants μ-sharded.  Two mesh axes have
    to move on the (μ, r) data axes; SPMD's all-to-all planner only
    handles one mesh axis at a time, so we stage through the cuSolverMp
    intermediate ``P(None, 'x', 'y')`` to keep every step a single-axis
    all-to-all primitive ``(a_X, b) → (a, b_X)``:

      Step 1  (q_, μ_, r_XY) → (q_, μ_X, r_Y)   ['x' moves r → μ]
      Step 2  (q_, μ_X, r_Y) → (q_, μ_XY, r_)   ['y' moves r → μ]
    """
    zeta = jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, 'x', 'y')))
    return jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, ('x', 'y'), None)))


def solve_zeta(
    L_q: jax.Array,
    Z_q: jax.Array,
    mesh_xy: Mesh,
    q_chunk_size: int = 1,
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    cct_trace_per_q: jax.Array | None = None,
) -> jax.Array:
    """
    Solve for zeta_q given pre-computed system matrix from
    :func:`factor_c_q`.

    For ``vertex_mu_L == 0`` ``L_q`` is the lower-triangular Cholesky
    factor of CCT and the inner solve is two triangular substitutions
    (``L y = Z`` then ``L^H ζ = y``).  This is the historical fast
    path — bit-identical to the previous implementation.

    For ``vertex_mu_L != 0`` ``L_q`` is the *unfactored* CCT^μ matrix.
    The transverse-channel CCT^μ is Hermitian but indefinite — γ̃^i ⊗
    γ̃^i has both signs of eigenvalue (eigenvalues of α^i are ±1), so
    Cholesky is invalid.  We solve via pivoted LU (``jnp.linalg.solve``)
    with a small diagonal ridge ``ε·|tr(L)|/n_rmu`` (ε = 1e-12) added
    to lift TRS-paired near-zero modes safely above the LU stability
    floor.  Bunch-Kaufman LDL^T would be the natural Hermitian-
    indefinite factorization but JAX doesn't expose it; pivoted LU is
    numerically equivalent for our purposes.

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
        solver_kind: 'auto' (default) defers to :func:`_resolve_solver_kind`;
                     explicit values are 'sharded_cholesky' (legacy 2D
                     blocked chol + per-q triangular solve), 'lu' (per-q
                     pivoted-LU for transverse channels),
                     'cusolvermp_cholesky' (distributed potrs via FFI),
                     or 'cusolvermp_lu' (distributed getrf+getrs via FFI
                     for the transverse channels).

    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, ('x','y'), None)
                — μ-axis flat-sharded across the ('x','y') mesh product,
                r-axis replicated.  This is the layout the downstream
                G-flat FFT (``accumulate_rchunk_to_gflat``) wants:
                each rank owns a μ-slab over the full r-extent, so the
                per-rank cuFFT runs locally without resharding.
    """
    nq, n_rmu, _ = L_q.shape
    _, _, n_zchunk = Z_q.shape

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L, solver_kind)

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrs: Z stays at P(None,'x','y'), no input reshard
        # and no all-gather of L.  Output reshards to P(None, ('x','y'),
        # None) so the downstream G-flat accumulator receives ζ
        # μ-flat-sharded (the FFT layout it actually wants — see
        # accumulate_rchunk_to_gflat / common.wfn_transforms).
        from ffi.cusolvermp import batched_distributed_potrs, CusolverMpBatchedLowerL
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # potrs requires Z's last dim divisible by Py; zero-pad columns
        # produce zero ζ columns and get trimmed on return.
        n_zchunk_padded = ((n_zchunk + Py - 1) // Py) * Py
        needs_padding = (n_zchunk_padded != n_zchunk)
        if needs_padding:
            Z_q = jnp.pad(Z_q, ((0, 0), (0, 0), (0, n_zchunk_padded - n_zchunk)),
                          mode='constant')
        # Re-attach handle metadata (the raw array carries no shape/grid info).
        L_handle = CusolverMpBatchedLowerL(
            raw=L_q, mesh=mesh_xy, n=int(n_rmu),
            mb=int(n_rmu) // Px, nb=int(n_rmu) // Py, nbatch=int(nq),
        )
        zeta_xy = batched_distributed_potrs(L_handle, Z_q, mesh=mesh_xy)
        # Natural potrs output sharding: P(None, 'x', 'y') = (q_, μ_X, r_Y).
        # Target: P(None, ('x','y'), None) = (q_, μ_XY, r_).
        # Single mesh axis 'y' moves from r-axis to μ-axis (joining 'x'
        # there) — one all-to-all on 'y' between data axes 1 and 2.
        zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta_xy, mesh_xy)
        if needs_padding:
            return zeta_out[:, :, :n_zchunk]
        return zeta_out

    if solver_kind == 'cusolvermp_lu':
        # Distributed getrf+getrs for the transverse channels.  L_q here
        # is the *unfactored* CCT^μ (Hermitian indefinite) — factor_c_q
        # passes it through.  Same input sharding, output reshard, and
        # column padding pattern as the cholesky branch.
        from ffi.cusolvermp import batched_distributed_solve_lu
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # getrs descB requires NRHS % Py == 0; zero-pad columns produce
        # zero ζ columns and get trimmed on return.
        n_zchunk_padded = ((n_zchunk + Py - 1) // Py) * Py
        needs_padding = (n_zchunk_padded != n_zchunk)
        if needs_padding:
            Z_q = jnp.pad(Z_q, ((0, 0), (0, 0), (0, n_zchunk_padded - n_zchunk)),
                          mode='constant')
        # Per-q ridge ε·|tr(L)|/n_rmu — same lift as the legacy 'lu'
        # branch, to keep TRS-paired near-zero modes above the LU
        # stability floor without perturbing well-conditioned ones.
        # ``cct_trace_per_q`` is precomputed once per channel by the
        # caller (fit_zeta_to_h5) — the trace doesn't change across
        # r-chunks since L_q is the per-channel CCT.  Computing it
        # inline here re-fires an all-reduce across the (μ_X, ν_Y)
        # sharding on every r-chunk: ~17 s GPU stream time on MoS2
        # 3×3 bispinor at our default chunk count.
        LU_RIDGE = 1e-12
        trace_per_q = (cct_trace_per_q if cct_trace_per_q is not None
                       else jnp.einsum('qii->q', L_q))
        ridge = (LU_RIDGE * jnp.abs(trace_per_q) / n_rmu)[:, None, None]
        eye_n = jnp.eye(n_rmu, dtype=L_q.dtype)[None, :, :]
        A_q = L_q + ridge * eye_n
        zeta_xy = batched_distributed_solve_lu(A_q, Z_q, mesh=mesh_xy)
        # Reshard rationale identical to the cholesky branch above.
        zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta_xy, mesh_xy)
        if needs_padding:
            return zeta_out[:, :, :n_zchunk]
        return zeta_out

    # Compute padding needed for even sharding across all devices
    total_devices = mesh_xy.devices.size
    n_zchunk_padded = ((n_zchunk + total_devices - 1) // total_devices) * total_devices
    needs_padding = n_zchunk_padded != n_zchunk

    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))
    L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
    q_batch = min(q_chunk_size, nq)
    nq_padded = ((nq + q_batch - 1) // q_batch) * q_batch

    # Dispatch on solver_kind (already resolved above) — independent of
    # vertex_mu_L so the rchunk kernel cache can collapse transverse
    # channels with the same solver_kind into a single compile.
    use_lu = (solver_kind == 'lu')
    # ``use_lu`` selects a pivoted-LU back-solve for transverse channels
    # (γ̃^i, i∈{1,2,3}).  CCT^μ for those channels is Hermitian but
    # indefinite (γ̃^i ⊗ γ̃^i has both signs of eigenvalue), so Cholesky
    # is invalid.  Bunch-Kaufman LDL^T would be the natural fit for
    # Hermitian indefinite, but JAX doesn't expose it; ``jnp.linalg.solve``
    # uses LU with partial pivoting which handles indefinite matrices
    # correctly as long as they aren't actually singular.  We keep a
    # small ridge ``LU_RIDGE·trace/n_rmu`` on the diagonal to lift any
    # near-zero modes from TRS-paired band cancellations safely above
    # the LU stability floor — small enough not to perturb the
    # well-conditioned modes.
    LU_RIDGE = 1e-12

    # Cache key for solve function (includes q_chunk_size and padded size).
    # ``use_lu`` partitions the cache so the Cholesky and LU compiles
    # don't collide on the same key.
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu,
                 n_zchunk_padded, q_chunk_size, bool(use_lu))

    def _ridge_indef_solve(L: jax.Array, Z: jax.Array) -> jax.Array:
        """Solve (L + ε·tr(L)/n · I) · ζ = Z via pivoted LU.

        ε = ``LU_RIDGE`` (1e-12).  The shift sits well below any
        physically meaningful eigenvalue but well above the partial-
        pivoting floor, so LU stays stable on TRS-paired near-zero
        modes without perturbing the rest of the spectrum.
        """
        n = L.shape[-1]
        ridge = LU_RIDGE * jnp.abs(jnp.trace(L)) / n
        L_reg = L + ridge * jnp.eye(n, dtype=L.dtype)
        return jnp.linalg.solve(L_reg, Z)

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            if use_lu:
                # Indefinite CCT^μ: pivoted-LU back-solve with ridge.
                return _ridge_indef_solve(L, Z_cols)
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
                # ``jnp.linalg.solve`` is natively batched on the leading
                # axis and dispatches one LU factorization per q.  Same
                # vmap structure as the Cholesky path so reshard plans
                # match.  We vmap the ridge-add per-q so each LU sees
                # its own conditioning shift.
                return jax.vmap(_ridge_indef_solve)(L_batch, Z_batch)
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
        # Two-step reshard via P('x',None,'y') intermediate.  Direct
        # P(None,'x','y') → P(None,None,('x','y')) moves both mesh axes
        # at once on the (μ, ν) data axes, which SPMD can't plan as a
        # single all-to-all and falls into Involuntary Full
        # Rematerialization (the full nq×μ×ν tensor materialised on
        # every device, ~16× per-device shard).  Staging through
        # P('x', None, 'y') parks 'x' on the leading nq axis so each
        # with_sharding_constraint moves only one mesh axis — two pure
        # all-to-alls, no all-gather inflation.  Same pattern as
        # w_isdf._get_w_solve_fn (V/χ reshard).  See lorrax_B commit
        # c0307a0 for the original Si 4×4×4 result (HLO peak
        # 68.94 → 29.94 GB on the same kernel compile).
        intermediate_shard = NamedSharding(mesh_xy, P('x', None, 'y'))
        @partial(jax.jit, donate_argnums=(0,))
        def _reshard_z(z):
            z = jax.lax.with_sharding_constraint(z, intermediate_shard)
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
        # Reshard r_XY → μ_XY so the downstream G-flat FFT runs
        # sharding-preserving (see _reshard_zeta_r_XY_to_mu_XY).
        result = _reshard_zeta_r_XY_to_mu_XY(result, mesh_xy)
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

    # Reshard r_XY → μ_XY so the downstream G-flat FFT runs
    # sharding-preserving (see _reshard_zeta_r_XY_to_mu_XY).
    zeta = _reshard_zeta_r_XY_to_mu_XY(zeta, mesh_xy)
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
    is_charge: bool = True,
    solver_kind: str = 'auto',
    q_irr_full_idx: np.ndarray | None = None,
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
    nk_tot = meta.nk_tot
    # In-memory shapes throughout this kernel use the PADDED μ extent.
    # ψ enters at padded (Phase 3a's load_centroids contract); all
    # bilinear consumers / WSCs / inner-jit boundary checks see
    # n_rmu_padded == ∏ p_a (mesh-divisible by construction).  L_q
    # is also at padded extent (factor_c_q embeds the
    # logical-extent factor with identity in the pad block — pad rows
    # of zeta come out as zero, logical block byte-identical to a
    # pure-logical solve).  meta.n_rmu (logical) is used only at the
    # SlabIO valid_shape= seam in fit_zeta_to_h5 so on-disk
    # extent stays logical and round-trips across mesh sizes.
    n_rmu = meta.n_rmu_padded
    nspinor = meta.nspinor
    kgrid = meta.kgrid

    # Closure-side IBZ row indices (Phase B).  None → full-BZ solve
    # (back-compat for ``write_ibz_only=False``).  Otherwise a static
    # jax int32 array baked into the jit's HLO.
    if q_irr_full_idx is not None:
        q_irr_idx_j = jnp.asarray(
            np.asarray(q_irr_full_idx, dtype=np.int32))
    else:
        q_irr_idx_j = None

    # ψ enters at PADDED n_rmu.  All in-memory arrays here operate at
    # PADDED extent so the inner shard_map boundaries see divisible
    # shapes (n_rmu_padded ≡ ∏ p_a).  The Cholesky factor L_q is at
    # PADDED extent too (factor_c_q embeds the logical-extent factor
    # with identity in the pad block); the back-solve produces zeta
    # with zero pad rows, logical block byte-identical to a logical-
    # only solve.  zeta is returned at padded extent; the SlabIO write
    # uses valid_shape=meta.n_rmu so on-disk extent stays logical.

    def z_q_phase(
        psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
        norms_l, norms_r, r_start_dyn, gamma_perm, gamma_phase,
    ):
        # Pre-multiply by 1/norms so the pair-density einsum sees the
        # norm-scaled input without a per-bc divide inside the scan
        # (algebraically identical: einsum is linear in psi_X·psi_Y).
        psi_l_X_scaled = psi_l_rmuT_X_fit / norms_l[None, None, :, None]
        psi_r_X_scaled = psi_r_rmuT_X_fit / norms_r[None, None, :, None]
        gamma_mu = None if is_charge else (gamma_perm, gamma_phase)
        # Round 6 streaming pair density + IFFT + γ̃·γ̃ + FFT — single
        # shard_map; io_callback pulls per-bc ψ(G) from the host store
        # inside the lax.scan body.  Output Z_q at FULL-BZ q-shape.
        return z_q_from_psi_sm(
            psi_l_X_scaled, psi_r_X_scaled, psi_G_store,
            band_chunk_ranges=band_chunk_ranges,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            r_start_dyn=r_start_dyn,
            r_chunk_size=actual_n_rchunk,
            gamma_L=gamma_mu,
            gamma_R=gamma_mu,
            kgrid=kgrid,
            mesh_xy=mesh_xy,
        )

    def solve_phase(Z_q, L_q, cct_trace_per_q):
        # IBZ-only solve (Phase B): L_q + cct_trace come in PRE-SLICED
        # at IBZ rows (via ``symmetry_maps.slice_q_full_to_ibz``
        # upstream in ``fit_zeta_to_h5``).  Z_q is built at full BZ by
        # ``z_q_phase``, so it gets the IBZ slice here.  When
        # ``q_irr_full_idx`` is None (write_ibz_only=False), all three
        # arrays are full-BZ and the solve runs as before.
        if q_irr_idx_j is not None:
            Z_q_for_solve = Z_q[q_irr_idx_j]
        else:
            Z_q_for_solve = Z_q
        return solve_zeta(
            L_q, Z_q_for_solve, mesh_xy, q_chunk_size,
            solver_kind=solver_kind,
            cct_trace_per_q=cct_trace_per_q)

    @jax.jit
    def _kernel(
        psi_l_rmuT_X_fit,
        psi_r_rmuT_X_fit,
        L_q,
        norms_l,
        norms_r,
        r_start_dyn,
        gamma_perm,
        gamma_phase,
        cct_trace_per_q,
    ):
        # Composed (z_q ∘ solve) under one ``@jax.jit`` — preserved for
        # the AOT memory model path which lowers a single callable.
        # Production ``fit_one_rchunk`` calls ``z_q_phase`` and
        # ``solve_phase`` directly with ``timing.section`` between, so
        # the per-r-chunk breakdown is host-visible.
        Z_q = z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn, gamma_perm, gamma_phase)
        return solve_phase(Z_q, L_q, cct_trace_per_q)

    # Attach the un-fused phases so ``fit_one_rchunk`` can time them.
    _kernel.z_q_phase = z_q_phase
    _kernel.solve_phase = solve_phase
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
    solver_kind: str = 'auto',
    q_irr_full_idx: np.ndarray | None = None,
    cct_trace_per_q: jax.Array | None = None,
):
    """Entry point for the r-chunk body jit.  Caches one compiled kernel
    per distinct static configuration.

    ``psi_G_store`` is captured in the jit closure (not a jit arg) so
    the compiled kernel calls its ``_slice_local_tile_bc`` method via
    io_callback inside ``z_q_from_psi_sm``'s scan body.  The cache key
    includes ``id(psi_G_store)`` to avoid reusing a compile built against
    a different store.
    """
    # vertex_mu_L splits into two runtime/closure pieces: a structural
    # ``is_charge`` (Python bool — keys the cache, separates the chol
    # branch from the gamma-fold branch) and the gamma matrix
    # ``(perm, phase)`` (runtime jit args — μ=1/2/3 all reuse the same
    # compiled HLO since the values never appear as static constants).
    is_charge = (int(vertex_mu_L) == 0)
    gamma_perm, gamma_phase = _gamma_perm_phase_mu(vertex_mu_L)
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
        bool(is_charge),
        str(solver_kind),
        (None if q_irr_full_idx is None
         else (int(q_irr_full_idx.shape[0]),
               hash(np.asarray(q_irr_full_idx,
                               dtype=np.int32).tobytes()))),
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
            is_charge=bool(is_charge),
            solver_kind=str(solver_kind),
            q_irr_full_idx=q_irr_full_idx,
        )
        _fit_one_rchunk_cache[cache_key] = fn
    # cct_trace_per_q is None for the charge channel (Cholesky path
    # ignores it); pass a tiny placeholder so the jit signature is
    # uniform across channels.  Size it to match L_q's q-axis (IBZ
    # extent when ``q_irr_full_idx`` is set; full BZ otherwise) — XLA
    # would dead-arg-eliminate it anyway, but keep the spec honest.
    if cct_trace_per_q is None:
        cct_trace_per_q = jnp.zeros((int(L_q.shape[0]),),
                                    dtype=jnp.complex128)

    # Call z_q and solve phases separately so each can be wrapped in
    # ``timing.section`` for per-r-chunk breakdown.  ``block_until_ready``
    # at each phase boundary is the cost of separating them — a few
    # microseconds on small kernels, dominated by the per-phase work.
    _dbg = bool(os.environ.get("LORRAX_RCHUNK_DEBUG"))
    _t_z0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.z_q_build"):
        Z_q = fn.z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn,
            gamma_perm, gamma_phase)
        Z_q.block_until_ready()
    _t_z = (time.perf_counter() - _t_z0) if _dbg else 0.0
    _t_s0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.solve"):
        zeta = fn.solve_phase(Z_q, L_q, cct_trace_per_q)
        zeta.block_until_ready()
    _t_s = (time.perf_counter() - _t_s0) if _dbg else 0.0
    if _dbg and jax.process_index() == 0:
        print(f"[rchunk_dbg]   z_q_build={_t_z*1000:.0f}ms "
              f"solve={_t_s*1000:.0f}ms", flush=True)
    return zeta


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
    n_avail = max(0, min(int(band_norms.shape[0]) - lo, hi - lo))
    sliced = np.zeros((hi - lo,), dtype=np.float64)
    if n_avail > 0:
        sliced[:n_avail] = np.asarray(band_norms[lo:lo + n_avail], dtype=np.float64)
    # max(1, 0) = 1 → padded entries divide ψ by 1, leaving the zeroed
    # ψ at zero.  No divide-by-zero hazard for [n_avail:nb] tail.
    return jnp.maximum(jnp.asarray(sliced), 1.0)


# ψ(G) no longer enters the jit as a tuple of arguments.  It lives on
# host, sharded n_XY over bands in per-rank tiles — each rank owns one
# contiguous ``(nk, nb/P, ns, nx, ny, nz)`` numpy array — and the
# kernel body slices per-bc (and optionally per-k) subsets via
# :func:`common.psi_G_store.PsiGStore._slice_local_tile_bc`.  See that module
# for the two lifecycle modes (host_cache vs file_reread).


def fit_zeta_to_h5(
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
    band_norms: np.ndarray | None = None,
    *,
    slab_io_backend=None,
    gspace_mode: str = "host_cache",
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    cusolvermp_charge: str = "auto",
    cusolvermp_lu: str = "auto",
    gflat_chunk_size: int = 0,
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
):
    """
    Full zeta fitting pipeline with r-chunk loop and HDF5 output.

    For ``vertex_mu_L == 0`` (default) this is the standard spin-traced
    path used by the charge-channel ISDF fit — bit-identical to the
    pre-bispinor implementation.  For ``vertex_mu_L ∈ {1, 2, 3}`` the
    pair-density helpers contract through the Lorentz vertex γ̃^{μ_L}
    instead of the identity, and ``factor_c_q`` /
    ``solve_zeta`` switch from Cholesky to LU because the
    transverse-channel CCT is indefinite.  See
    ``docs/BISPINOR_DHFB_DESIGN.md`` for the math.

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
    from gw.gw_config import SlabIOBackend
    if slab_io_backend is None:
        slab_io_backend = SlabIOBackend.H5PY_ALLGATHER
    # Treat both per-rank-parallel-write PHDF5 backends (FFI on GPU,
    # mpi_host on CPU) as the "fast write" path; only H5PY_ALLGATHER
    # uses the rank-0-gather code below.
    use_ffi_io = slab_io_backend in (
        SlabIOBackend.PHDF5_FFI, SlabIOBackend.PHDF5_HOST)

    # P0 — entry of ζ-fit.  Captures the persistent state set up by
    # ``prepare_isdf_and_wavefunctions`` BEFORE ζ-fit starts: ψ at
    # centroids (full [b0, b4) band range, both Y and X transposes),
    # gflat_acc allocation will not have happened yet.  Forms the
    # planner's "Peak C const" baseline.  Round-1 addition.
    mem_probe("zeta_fit_start")

    nx, ny, nz = meta.fft_grid
    # Two μ extents flow through this function (see common/meta.py:38):
    # ``n_rmu`` is the LOGICAL centroid count from the centroid file;
    # ``n_rmu_padded`` rounds up to ``world_size = ∏ p_a`` so any
    # single- or product-axis sharding on the μ dim divides cleanly.
    # ψ is delivered at PADDED extent by ``load_centroids_band_chunked``
    # (Phase 3a) — pad rows zero — and stays there through the
    # in-memory pair-density / CCT chain.  The Cholesky in
    # ``factor_c_q`` slices internally to logical via the
    # ``n_rmu_logical=`` kwarg (Phase 3b-Cholesky) so the factorization
    # sees a non-singular matrix at its true extent.  zeta_q on disk
    # has logical extent (SlabIO ``valid_shape=`` clips the padded
    # output before write).
    n_rmu = meta.n_rmu                      # logical
    n_rmu_padded = meta.n_rmu_padded        # padded
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
    # γ̃^0 = I_4 → vertex_mu_L=0 is the standard spin-traced path.  For
    # vertex_mu_L ∈ {1,2,3} the γ̃^μ vertex is folded into both P_l and
    # P_r so C_q is the proper per-channel interpolation metric for the
    # Lorentz pair density.  CCT^μ for transverse channels is Hermitian
    # indefinite and rank-deficient: TRS in non-magnetic ground states
    # gives near-null transverse-current modes that would be amplified
    # by 10^4–10^6 if we naively LU-solved through them (the original
    # MoS2 σ^B blowup).  The robust solver in :func:`solve_zeta`
    # uses an SVD pseudoinverse with rcond cutoff to drop those null
    # modes instead of inverting through them — the unique min-norm LSQ
    # solution.
    # Force-eager-import gamma_matrices so its module-level
    # ``gammas_sparse = [_to_sparse(g) for g in gammas]`` (which calls
    # ``jnp.nonzero``) runs OUTSIDE any JIT trace; otherwise the first
    # reference comes from inside the per-chunk kernel jit and trips
    # a ConcretizationTypeError.
    if int(vertex_mu_L) != 0:
        from . import gamma_matrices as _gm  # noqa: F401  (warm import)

    # ── Finalize write_ibz_only BEFORE any IBZ slicing (bug fix) ─────────
    # The IBZ cascade slices C_q/L_q to IBZ rows in STEP 2/3 below, and
    # slices Z_q to IBZ inside the per-r-chunk kernel; the two MUST agree.
    # The orbit-closure auto-fallback can flip write_ibz_only=False when the
    # centroid set isn't closed under the WFN sym group, so it must run HERE
    # — before the C_q slice.  (Previously it ran after factor_c_q, so the
    # charge channel sliced L_q to IBZ, then fell back, leaving L_q at IBZ
    # while Z_q stayed full-BZ → the ``B.shape[0]=nq_full != Nq=nq_ibz``
    # distributed-potrs crash.)  Transverse channels can't fall back (the
    # V_q orchestrator assumes IBZ ζ̃_T), so they loud-fail with a hint.
    if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
        try:
            from centroid.orbit_syms import (
                compute_centroid_sym_perm as _check_perm,
            )
            _cent_idx_for_check = np.asarray(
                jax.device_get(centroid_indices), dtype=np.int32)
            _ntran_check = int(np.asarray(sym.sym_matrices).shape[0])
            # ``sym.sym_matrices`` holds the spatial ops; the fractional
            # translations live on WFNReader (BGW WFN.h5 layout).
            _check_perm(
                _cent_idx_for_check,
                sym_matrices=np.asarray(sym.sym_matrices[:_ntran_check]),
                translations=np.asarray(wfn.translations[:_ntran_check]),
                fft_grid=np.asarray(meta.fft_grid, dtype=np.int32),
            )
        except RuntimeError as _exc:
            _first = (_exc.args[0].splitlines()[0]
                      if _exc.args else str(_exc))
            if int(vertex_mu_L) != 0:
                raise RuntimeError(
                    f"Bispinor transverse zeta_T (mu_L={int(vertex_mu_L)}) "
                    f"IBZ-write requested, but the transverse centroid set "
                    f"fails the orbit-closure check under the WFN sym group: "
                    f"{_first}.  Regenerate the transverse centroid file with "
                    f"``centroid.kmeans_cli --density-mode current`` "
                    f"(orbit-aware by default for ntran>1) so the set is "
                    f"closed under the spatial sym group, or bypass the "
                    f"bispinor IBZ cascade with ``LORRAX_FORCE_FULL_BZ=1``."
                ) from _exc
            if jax.process_index() == 0:
                print(f"  q-IBZ reduction: centroid orbit closure failed "
                      f"— falling back to full-BZ on disk.  Reason: {_first}")
            write_ibz_only = False

    with timing.section("zeta_fit.CCT"):
        # ψ inputs at PADDED n_rmu (Phase 3a's load_centroids contract).
        # Monolithic shard_map pipeline: open-spin pair density + IFFT
        # + γ̃·γ̃ + FFT fused inside one shard_map.  The rank-5
        # P_l/P_r pair density never exists as a global XLA value, so
        # the rank-3 fused-replicated reshape that pegged the kernel
        # peak under the legacy chain cannot form.  γ̃^μ_L applied at
        # the post-IFFT contraction step (charge: identity short-
        # circuit; transverse: (perm, phase) tuple).  Output C_q is
        # rank-3 (k, μ, ν).
        chan_label = ("charge γ̃^0=I" if vertex_mu_L == 0
                      else f"transverse γ̃^{vertex_mu_L}")
        print(f"  Computing C_q via shard_map pipeline (open-spin, {chan_label})")
        if vertex_mu_L == 0:
            C_q = c_q_from_psi_sm(
                psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                kgrid=kgrid, mesh_xy=mesh_xy)
        else:
            gamma_mu = _gamma_perm_phase_mu(vertex_mu_L)
            C_q = c_q_from_psi_sm(
                psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                gamma_mu, gamma_mu,
                kgrid=kgrid, mesh_xy=mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu_padded, n_rmu_padded) with zero
        # pad rows/cols.

        # Flatten for Cholesky.  Reshape uses padded extent (the
        # in-memory shape); factor_c_q slices to logical
        # internally via ``n_rmu_logical=``.
        C_q_flat = C_q.reshape(nq, n_rmu_padded, n_rmu_padded)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

        # IBZ cascade for the per-q factor: slice C_q to IBZ rows *before*
        # ``factor_c_q`` runs so Cholesky / LU factors only ``n_q_ibz``
        # blocks instead of all ``n_q_full``.  C_q has the same (n_q, μ, ν)
        # shape as V_q, and Cholesky is per-q independent — slice-then-
        # factor gives bit-equal L_q rows as factor-then-slice.  The
        # downstream solve still produces ζ_q at IBZ, and V_q unfolds via
        # ``common.symmetry_maps.unfold_v_q`` from IBZ → full BZ.  Same
        # slice helper applies to χ_q for the W_q = (1 − v_q χ_q)^{-1} v_q
        # path once that lands.
        if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
            from .symmetry_maps import slice_q_full_to_ibz
            C_q_flat = slice_q_full_to_ibz(
                C_q_flat, sym.q_irr_full_idx, out_sharding=flat_shard)

    # ========== STEP 3: Compute L_q from CCT ==========
    # μ_L=0 (charge): C_q is PSD → 2D-blocked Cholesky factor L_q.
    # μ_L=1,2,3 (transverse): C_q is Hermitian indefinite — skip the
    # factorization and pass the slice through; the per-chunk
    # solve_zeta dispatches to an SVD pseudoinverse with
    # rcond cutoff (drops null transverse-current modes that would
    # otherwise be amplified by 10^4–10^6).
    with timing.section("zeta_fit.cholesky"):
        # Resolve once so the banner reflects what actually runs and
        # downstream callees skip their own 'auto' fallback.
        _resolved_solver_kind = _resolve_solver_kind(
            mesh_xy, int(vertex_mu_L), solver_kind,
            cusolvermp_charge=cusolvermp_charge,
            cusolvermp_lu=cusolvermp_lu)
        if int(vertex_mu_L) == 0:
            print(f"  Computing L_q = chol(C_q)  [PSD, charge channel, "
                  f"path={_resolved_solver_kind}]")
        else:
            print(f"  Pass through C_q  [γ̃^{vertex_mu_L} indefinite — "
                  f"path={_resolved_solver_kind}]")
        L_q = factor_c_q(
            C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
            n_rmu_logical=n_rmu, solver_kind=_resolved_solver_kind)
        L_q.block_until_ready()
        print(f"  L_q: {L_q.shape}")

    # Pre-compute per-q trace of L_q ONCE per channel.  Only the
    # transverse (LU) path uses it (for the ridge ``ε·|tr(L)|/n_rmu``
    # before each per-q LU solve).  Computing inside solve_zeta means an
    # all-reduce across the (mu/p_x, mu/p_y) mesh sharding fires on every
    # r-chunk — 17 s of GPU stream time on MoS2 3×3 bispinor across 4
    # r-chunks × 3 transverse channels.  L_q (which is CCT for the LU
    # path) doesn't change across r-chunks, so the trace is invariant.
    if int(vertex_mu_L) != 0:
        with timing.section("zeta_fit.trace_L_q"):
            cct_trace_per_q = jnp.einsum('qii->q', L_q)
            cct_trace_per_q.block_until_ready()
    else:
        cct_trace_per_q = None

    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        gc.collect()
        jax.clear_caches()  # Clear JAX function caches that may hold array refs

    # ========== STEP 4a: q-IBZ reduction + header writes (rank 0) ==========
    # When ``write_ibz_only=True`` (default), ζ is written for IBZ q's
    # only.  V_q at the full BZ is recovered by the reader / V_q
    # orchestrator using sym data from ``mf_header`` (see report.md
    # §2.4).  The on-disk ``zeta_q`` leading axis is ``n_q_disk``
    # rather than ``n_q_full``; the chunk loop slices
    # ``zeta_chunk[q_irr_full_idx]`` before writing.
    #
    # When ``write_ibz_only=False`` (caller forced full-BZ writes via
    # ``LORRAX_FORCE_FULL_BZ=1``), the full-BZ axis is preserved on
    # disk for back-compatibility.
    #
    # ``write_ibz_only`` was finalized above (before the C_q/L_q IBZ slice)
    # by the orbit-closure auto-fallback, so the on-disk q-axis is IBZ when
    # it is True and full-BZ when it fell back — nothing more to decide here.

    # BGW Brillouin-zone wrap used by the V_q kernel
    # (``_qvec_wrap`` at ``gw/v_q_tile.py:1204``): ``q > kgrid/2 → q
    # − kgrid``.  The writer must match so the per-q phase
    # ``exp(-2πi (q/kgrid)·r)`` baked into the G-flat output is the
    # convention the consumer expects.
    def _bgw_wrap_q(q_int_kgrid: np.ndarray) -> np.ndarray:
        kg = np.asarray(meta.kgrid, dtype=np.float64)
        q = np.asarray(q_int_kgrid, dtype=np.float64)
        return np.where(q > kg / 2, q - kg, q)

    if write_ibz_only:
        q_irr_kgrid_int = sym.q_irr_kgrid_int
        q_irr_full_idx = sym.q_irr_full_idx
        n_q_disk = int(q_irr_full_idx.shape[0])
        # IBZ fractional q-vectors for the G-flat accumulator (Phase C1b).
        # BGW wrap THEN divide by kgrid so the writer's per-q phase
        # matches the V_q kernel's ``apply_bloch_phase`` convention.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(q_irr_kgrid_int)
                       / _kgrid_arr_for_qfrac[None, :])
        print(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
              f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        q_irr_frac = None
        n_q_disk = nq
        print(f"  q axis on disk: full BZ ({nq} q-points) "
              f"(write_ibz_only=False or closure check failed)")

    # ---- G-flat on-disk format ---------------------------------
    # The writer accumulates each r-chunk's contribution into a
    # persistent G-flat buffer via
    # ``common.wfn_transforms.accumulate_rchunk_to_gflat`` and writes
    # the final tensor as ``zeta_q_G`` (shape
    # ``(n_q_disk, n_rmu, ngkmax)``).  The full r-space ζ_q is never
    # materialised on disk or as a persistent device buffer.  When
    # ``zeta_cutoff_ry`` is provided we build the per-q WFN.h5-style
    # sphere ``{G : |q+G|² ≤ cutoff}``, pad to a uniform ``ngkmax``
    # with the sentinel Miller index ``(-nx/2, -ny/2, -nz/2)``, and
    # store both the coeffs and the per-q components on disk.  Without
    # a cutoff the writer falls back to the full flat-FFT axis
    # (n_G_sph = n_rtot) — slow disk path, kept for sanity checks.
    if q_irr_frac is None:
        # Full-BZ q-vectors with BGW wrap, then / kgrid — same convention
        # the V_q kernel's ``_zeta_disk_to_G`` consumed via
        # ``_qvec_wrap``.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(sym.kvecs_asints)
                       / _kgrid_arr_for_qfrac[None, :])

    # Build the per-q WFN.h5-style sphere when a cutoff is available.
    # The output is host numpy; the writer threads ``sphere_idx_padded``
    # through ``accumulate_rchunk_to_gflat`` and stashes the components
    # / ngk / cutoff into the isdf_header below.  ``zeta_cutoff_ry``
    # — distinct from V_q's bare-Coulomb cutoff — defines the per-q
    # sphere on disk.  Caller (``gw_init.fit_zeta``) validates
    # ``zeta_cutoff_ry ≥ bare_coulomb_cutoff_ry`` so V_q has every G
    # it needs.
    _gflat_sphere_idx_padded = None      # (n_q_disk, ngkmax) int32
    _gflat_gvec_components = None        # (n_q_disk, 3, ngkmax) int32
    _gflat_ngk_per_q = None              # (n_q_disk,) int32
    _gflat_ngkmax = None
    if zeta_cutoff_ry is not None and int(meta.sys_dim) != 0:
        from common.coulomb_sphere import compute_per_q_bare_coulomb_components
        _bvec_for_sphere = np.asarray(
            wfn.blat * wfn.bvec, dtype=np.float64)
        _sphere_pkg = compute_per_q_bare_coulomb_components(
            fft_grid=meta.fft_grid,
            bvec=_bvec_for_sphere,
            q_irr_frac=q_irr_frac,
            vcoul_cutoff_ry=float(zeta_cutoff_ry),
            sys_dim=int(meta.sys_dim),
        )
        _gflat_sphere_idx_padded = _sphere_pkg["sphere_idx_padded"]
        _gflat_gvec_components = _sphere_pkg["gvec_components_padded"]
        _gflat_ngk_per_q = _sphere_pkg["ngk_per_q"]
        _gflat_ngkmax = int(_sphere_pkg["ngkmax"])
        if jax.process_index() == 0:
            print(
                f"  G-flat ζ sphere: ngkmax={_gflat_ngkmax}, "
                f"min ngk={int(_gflat_ngk_per_q.min())}, "
                f"max ngk={int(_gflat_ngk_per_q.max())} "
                f"({_gflat_ngkmax / float(n_rtot):.3%} of n_rtot)")

    # ``zeta_q.h5`` carries the BGW-style ``mf_header`` verbatim from
    # the source WFN so any downstream consumer (the new
    # :class:`file_io.zeta_reader.ZetaReader`, or anything else that
    # speaks the WFN.h5 header) sees the same crystal / k-grid / G-grid
    # / symmetry view.  ``isdf_header`` holds ζ-specific metadata only
    # — centroids in FFT-grid + fractional coords, density label,
    # ``vertex_mu_L``.  Everything sym-derivable (q-IBZ list, centroid
    # orbit permutation, G-sphere) is rebuilt at read time via
    # ``SymMaps`` + ``orbit_syms`` and is *not* stored.
    #
    # Sequence: rank 0 pre-stripes the file, writes both header groups
    # in mode='w' (truncate), closes.  Then SlabIO re-opens with
    # mode='a' so the headers survive and ``create_dataset('zeta_q')``
    # appends rather than truncates.
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    from file_io._slab_io_ffi import _lustre_prestripe

    _wfn_src_path = getattr(wfn, '_filename', None)
    if _wfn_src_path is None:
        raise ValueError(
            "fit_zeta_to_h5: wfn must expose '_filename' (the source "
            "WFN.h5 path) so mf_header can be copied verbatim into "
            "zeta_q.h5.")

    # Centroid FFT-grid indices for the isdf_header.  ``centroid_indices``
    # may be a jax.Array on device; pull to host as int32 (n_rmu, 3).
    _cent_idx_np = np.asarray(jax.device_get(centroid_indices),
                              dtype=np.int32)
    if _cent_idx_np.shape != (n_rmu, 3):
        raise ValueError(
            f"fit_zeta_to_h5: centroid_indices has shape "
            f"{_cent_idx_np.shape}, expected ({n_rmu}, 3).")
    _density_label = 'scalar' if int(vertex_mu_L) == 0 else 'current'
    _hdr_kwargs = dict(
        r_mu_fft_idx=_cent_idx_np,
        fft_grid=meta.fft_grid,
        density=_density_label,
        vertex_mu_L=int(vertex_mu_L),
        zeta_layout='G_flat',
    )
    if _gflat_gvec_components is None:
        raise ValueError(
            "G-flat ζ writer requires a ζ sphere — pass "
            "zeta_cutoff_ry to fit_zeta_to_h5.")
    _hdr_kwargs.update(
        gvec_components=_gflat_gvec_components,
        ngk_per_q=_gflat_ngk_per_q,
        zeta_cutoff_ry=float(zeta_cutoff_ry),
    )
    _isdf_hdr = IsdfHeader.build(**_hdr_kwargs)

    with timing.section("zeta_fit.write_headers"):
        if jax.process_index() == 0:
            # Pre-stripe the file (delete + lfs setstripe).  Idempotent
            # no-op on non-Lustre filesystems.  Must happen before any
            # h5py create so the stripe layout survives ``H5Fcreate``.
            stripe_count = int(
                os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get(
                "LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(output_file, stripe_count=stripe_count,
                              stripe_size=stripe_size)
            # Create file with mf_header, then append isdf_header.
            copy_mf_header(_wfn_src_path, output_file, dst_mode='w')
            write_isdf_header(output_file, _isdf_hdr, mode='a')
        jax.experimental.multihost_utils.sync_global_devices(
            "zeta_fit_headers_written")

    # ========== STEP 4b: SlabIO appends zeta_q to the pre-created file ==========
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
    #
    # mode='a' (not 'w') so the pre-written mf_header + isdf_header
    # are preserved.  SlabIO's FFI prestripe step is skipped on 'a'
    # — we already striped above.
    #
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
        # G-flat layout: ``zeta_q_G`` dataset (n_q_disk, n_rmu, ngkmax)
        # — WFN.h5 ``wfns/coeffs`` style with a fixed ``ngkmax`` padded
        # G axis.  Per-q components live in
        # ``isdf_header/gvec_components`` (already serialised by the
        # write_isdf_header call above).  Chunking: one row per q ×
        # full μ × full ngkmax keeps per-q reads contiguous.
        _n_G_sph = (int(_gflat_ngkmax)
                     if _gflat_ngkmax is not None else n_rtot)
        if use_ffi_io:
            zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                             backend=slab_io_backend)
            zeta_io.create_dataset(
                'zeta_q_G',
                shape=(n_q_disk, n_rmu, _n_G_sph),
                dtype=np.complex128,
                chunks=(1, n_rmu, _n_G_sph),
            )
        else:
            with SlabIO(output_file, mode='a', mesh=mesh_xy,
                        backend=slab_io_backend) as _zeta_create_io:
                _zeta_create_io.create_dataset(
                    'zeta_q_G',
                    shape=(n_q_disk, n_rmu, _n_G_sph),
                    dtype=np.complex128,
                    chunks=(1, n_rmu, _n_G_sph),
                )
            zeta_io = None

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # ``gspace_mode`` (cohsex.in ``gspace_mode``; see
    # ``GspaceIO`` enum): ``host_cache`` is the default; ``file_reread``
    # rebuilds the per-rank host ψ(G) buffer at each r-chunk for
    # multi-TB WFN.h5 systems that can't hold ψ(G) resident.

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
        wfn=wfn, sym=sym, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        bispinor=bispinor,
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

    # ``LORRAX_MEM_DEBUG=1`` — runtime probe of process-wide HBM at
    # named lifecycle sites.  The module-level ``mem_probe`` helper is
    # reused so the r-chunk loop sites and the gw_init V_q sites all
    # share one source of truth.  HLO's buffer-assignment.txt is per-jit
    # and cannot prove cross-jit liveness — see
    # reports/memory_model_refit_2026-05-17/agent_e_cross_jit_lifetime.md.
    _mem_probe = mem_probe

    # Per-chunk: ``accumulate_rchunk_to_gflat`` adds the chunk's
    # contribution into the donated ``gflat_acc`` in place; no
    # per-chunk SlabIO write.  The single ``zeta_q_G`` write happens
    # once after the loop.

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

    # ---- G-flat accumulator (zero-init, μ-sharded) ----
    # Persistent buffer: (n_q_disk, n_rmu_padded, ngkmax) c128 with
    # μ sharded across ('x', 'y') so each rank holds n_rmu/p per q.
    # Donated to ``accumulate_rchunk_to_gflat`` each iter; in-place add.
    # When the per-q sphere isn't available (no vcoul_cutoff_ry) we
    # fall back to the full flat-FFT axis n_rtot — slow, kept for
    # smoke / sanity tests.
    from common.wfn_transforms import accumulate_rchunk_to_gflat
    # μ allocated at PADDED extent so the ('x','y') sharding divides
    # cleanly.  Pad rows are zero because the back-solve produces
    # zeta_pad = 0 (L_q's pad block is identity).
    _n_rmu_padded = int(meta.n_rmu_padded)
    _gflat_acc_n_G = (int(_gflat_ngkmax)
                       if _gflat_ngkmax is not None else n_rtot)
    _gflat_acc_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    gflat_acc = jax.jit(
        lambda: jnp.zeros(
            (n_q_disk, _n_rmu_padded, _gflat_acc_n_G),
            dtype=jnp.complex128),
        out_shardings=_gflat_acc_sharding,
    )()
    # Flat-axis chunking inside ``accumulate_rchunk_to_gflat``.  The
    # kernel runs inside a ``shard_map`` over ``('x','y')`` and chunks
    # the per-rank flat ``(n_q · n_mu_local)`` axis into rows-per-
    # scan-iteration of ``chunk_size``.  Memory bound:
    # ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.
    #
    # ``gflat_chunk_size = 0`` ⇒ one-shot (fine when the full per-rank
    # box ``N · n_rtot · 16 B`` fits; MoS2 3×3 at 4 ranks: 1.1 GB).
    # For CrI3-class FFT grids set cohsex.in ``gflat_chunk_size`` to
    # an integer; the kernel zero-pads N up to a multiple of the chunk
    # size so any value works (no divisibility constraint on either
    # n_q or n_mu_local).
    _gflat_chunk_size = int(gflat_chunk_size) if gflat_chunk_size else None
    if jax.process_index() == 0:
        _p_prod = int(jax.device_count())
        _n_mu_local = int(meta.n_rmu_padded) // _p_prod
        _N = n_q_disk * _n_mu_local
        _cs = _gflat_chunk_size or _N
        print(f"  G-flat ζ accumulator: N={_N} rows/rank "
              f"(n_q={n_q_disk} × n_mu_local={_n_mu_local}); "
              f"chunk_size={_cs} → "
              f"per-iter FFT box {_cs * n_rtot * 16 / 1e9:.2f} GB/rank")
    # Numpy → replicated: avoid the ``jnp.asarray`` wrap that would
    # single-device-stage and turn device_put into an all-reduce.
    _q_irr_frac_dev = jax.device_put(
        np.asarray(q_irr_frac, dtype=np.float64),
        NamedSharding(mesh_xy, P(None, None)))

    # P1 — pre r-chunk loop, after L_q computed AND gflat_acc allocated.
    # This is the persistent baseline the planner's ``_peak_C_const``
    # should match: centroids (ψ_l/ψ_r in both Y and X transposes), L_q
    # (Cholesky factor at IBZ for charge / pass-through CCT for
    # transverse), and the freshly-zeroed gflat_acc.  Round-1 addition.
    if os.environ.get("LORRAX_MEM_DEBUG"):
        jax.block_until_ready(gflat_acc)
        jax.block_until_ready(L_q)
    mem_probe("pre_rchunk_loop")

    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start

            # file_reread mode: (re)build the host-side ψ(G) tiles
            # for this r-chunk.  host_cache mode: no-op.
            psi_G_store.begin_rchunk(r_start, r_end)

            _dbg_rchunk = bool(os.environ.get("LORRAX_RCHUNK_DEBUG"))
            _mem_probe(f"rchunk_start chunk={chunk_idx}")
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
                        vertex_mu_L=int(vertex_mu_L),
                        solver_kind=_resolved_solver_kind,
                        q_irr_full_idx=q_irr_full_idx,   # Phase B: gather inside the kernel
                        cct_trace_per_q=cct_trace_per_q,
                    )
                    zeta_chunk.block_until_ready()
            finally:
                # MUST run after block_until_ready — under file_reread
                # the host tiles are freed here and any still-pending
                # io_callback would use-after-free.
                psi_G_store.end_rchunk()
            _t_fit = time.perf_counter() - t0
            t_fit_total += _t_fit
            _mem_probe(f"after_fit_one_rchunk chunk={chunk_idx}")

            # 6e. IBZ-slice → allgather (or FFI) → HDF5 write.
            # ``zeta_chunk`` is computed at full BZ q (the FFT in
            # ``solve_zeta`` naturally outputs all q's).  We slice to
            # Phase B: ``zeta_chunk`` is already IBZ-shape
            # (n_q_disk, n_rmu, n_rchunk) — the gather happens inside
            # ``fit_one_rchunk`` before the triangular solve.  In
            # full-BZ mode (q_irr_full_idx=None) the kernel returns
            # full-BZ shape.  Accumulate this r-chunk's contribution
            # into ``gflat_acc`` in place; the full ``zeta_q_G`` is
            # written once after the loop.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                gflat_acc = accumulate_rchunk_to_gflat(
                    rchunk=zeta_chunk, gflat_acc=gflat_acc,
                    fft_grid=meta.fft_grid, r0=r_start,
                    sphere_idx=_gflat_sphere_idx_padded,
                    qvec_frac=_q_irr_frac_dev,
                    norm='backward',
                    chunk_size=_gflat_chunk_size,
                    mesh=mesh_xy,
                )
                del zeta_chunk
                if os.environ.get("LORRAX_MEM_DEBUG"):
                    jax.block_until_ready(gflat_acc)
            _t_write = time.perf_counter() - t0
            t_write_total += _t_write
            _mem_probe(f"after_accumulate chunk={chunk_idx}")
            if _dbg_rchunk and jax.process_index() == 0:
                print(f"[rchunk_dbg] chunk={chunk_idx+1}/{num_chunks} "
                      f"r=[{r_start},{r_end}) fit={_t_fit*1000:.0f}ms "
                      f"write={_t_write*1000:.0f}ms "
                      f"total={(_t_fit+_t_write)*1000:.0f}ms", flush=True)
            r_progress.step()
            # LORRAX_MAX_RCHUNKS=N: stop the r-chunk loop after N chunks
            # for profiling/sweeping.  Clean python exit avoids the
            # SLURM step-zombie issue you get from killing the python
            # mid-run.  Off when unset.
            _max_rchunks = os.environ.get("LORRAX_MAX_RCHUNKS")
            if _max_rchunks and (chunk_idx + 1) >= int(_max_rchunks):
                if jax.process_index() == 0:
                    print(f"[rchunk_dbg] LORRAX_MAX_RCHUNKS={_max_rchunks} "
                          f"reached after chunk {chunk_idx+1}; "
                          f"breaking r-chunk loop for profiling.",
                          flush=True)
                break


    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    # Sample GPU memory ONCE after the last chunk's jit settles.  The
    # allocator keeps the peak reservation so this reads close to the
    # all-time high water.
    _track_peak()

    # ---- Write the accumulated G-flat ζ_q ----
    # One collective write of the persistent ``(n_q_disk, n_rmu,
    # ngkmax)`` tensor to disk.
    with timing.section("zeta_fit.write_g_flat"):
        # Pad slot zero-fill (WFN.h5 ``coeffs = 0`` convention).  The
        # per-q gather inside ``accumulate_rchunk_to_gflat`` read the
        # sentinel ``(-nx/2, -ny/2, -nz/2)`` flat-FFT slot into every
        # pad position; those values are physical (not zero) so we
        # mask them here.  Logical slots ``[..., :ngk[q]]`` carry the
        # real coeffs and are untouched.
        if _gflat_ngk_per_q is not None:
            _ngk_dev = jax.device_put(
                np.asarray(_gflat_ngk_per_q, dtype=np.int32),
                NamedSharding(mesh_xy, P(None)))
            _g_axis = jnp.arange(int(gflat_acc.shape[-1]),
                                  dtype=jnp.int32)        # (ngkmax,)
            _mask = (_g_axis[None, None, :] < _ngk_dev[:, None, None])
            gflat_acc = jnp.where(
                _mask, gflat_acc, jnp.zeros_like(gflat_acc))
        jax.block_until_ready(gflat_acc)
        _n_G_sph = int(gflat_acc.shape[-1])
        if use_ffi_io:
            # On-disk extent is LOGICAL n_rmu; in-memory buffer is
            # PADDED ``n_rmu_padded``.  SlabIO ``valid_shape=`` clips
            # the trailing μ pad rows on write (they are zero by
            # construction — L_q's pad block is identity).
            zeta_io.write_slab(
                'zeta_q_G', gflat_acc,
                offset=(0, 0, 0),
                global_shape=(n_q_disk, n_rmu, _n_G_sph),
                valid_shape=(n_q_disk, n_rmu, _n_G_sph),
            )
        else:
            # allgather backend: one per-q gather (not per chunk).
            # The full tensor is at most a few GB replicated; for
            # CrI3 scale the FFI backend is mandatory anyway.
            from file_io._slab_io_allgather import _to_host as _gather_to_host
            _g = _gather_to_host(gflat_acc)
            if jax.process_index() == 0:
                import h5py as _h5
                with _h5.File(output_file, 'a') as _f:
                    _f['zeta_q_G'][...] = _g
            del _g
    del gflat_acc

    # Close the SlabIO handle (FFI path only; allgather path never
    # opened one after STEP 4).
    with timing.section("zeta_fit.close_io"):
        if zeta_io is not None:
            zeta_io.close()

    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Flip ``isdf_header/zeta_is_done`` to True now that every chunk
    # has drained to disk.  Restart paths key off this flag to decide
    # whether the on-disk ζ is trustable; flipping it here (after the
    # global sync above) guarantees every rank's writes are durable.
    if jax.process_index() == 0:
        from file_io.isdf_header import mark_zeta_done
        mark_zeta_done(output_file)

    # Free the host tiles (host_cache mode only; file_reread's tiles
    # are already empty after the final end_rchunk).  The phdf5 reader
    # itself is cached at module level and survives.
    psi_G_store.close()

    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    print(f"  Zeta output: {output_file}  shape: "
          f"(n_q_disk={n_q_disk} of {nqx}·{nqy}·{nqz}={nq} full-BZ, "
          f"n_rtot={n_rtot}, n_rmu={n_rmu})")
    print(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

    # P3 — exit of ζ-fit.  Captures what's still alive after the chunk
    # loop completes: gflat_acc was del'd above, zeta_chunk freed, but
    # centroids (psi_l/psi_r) and L_q are still referenced by the
    # caller's closure (they were passed in as args).  V_q runs next
    # against this baseline.  Round-1 addition.
    mem_probe("zeta_fit_end")

    # Return only peak-memory high-water mark; centroid wavefunctions
    # are not returned (see docstring — callers re-load them directly
    # via ``load_centroids_band_chunked``).
    return _peak_bytes

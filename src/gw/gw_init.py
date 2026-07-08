"""ISDF fitting orchestration for LORRAX GW.

  fit_zeta / compute_V_q / build_wavefunction_bundle — pipeline steps
  prepare_isdf_and_wavefunctions — top-level orchestrator called by main()

Chunk sizing (band_chunk / r_chunk / q_chunk / gflat_chunk_size) is owned
entirely by :func:`gw.gflat_memory_model.plan_gflat_chunks` — the single
production planner (persistent floor + max over five stage transients).
"""
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import h5py

import common.timing as timing
from common import jax_profile



def get_effective_chunk_size(chunk_size: int) -> int | None:
    """Convert chunk_size flag: -1=None (all bands), 0=auto (64), 1-2048=explicit."""
    if chunk_size == -1:
        return None
    if chunk_size == 0:
        return 64
    if 1 <= chunk_size <= 2048:
        return chunk_size
    raise ValueError(f"chunk_size must be -1, 0, or 1-2048, got {chunk_size}")


# Backward-compatible re-exports
from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401


def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}.

	Legacy helper used by psp/get_DFT_mtxels.py.  GW code uses BandSlices instead.
	"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
             psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print):
	"""Fit ISDF interpolation vectors ζ and write to HDF5.

	The caller supplies (a) the full-range centroid wavefunctions
	(``psi_rmu_Y`` / ``psi_rmuT_X``, spanning [b0, b4) as returned by
	``load_centroids_band_chunked``) and (b) the chunk plan dict from
	:func:`gw.gflat_memory_model.plan_gflat_chunks`.  Returns
	``(zeta_h5_path, mem_est, transverse_wfn_data)``.
	"""
	from gw.isdf_fitting import fit_zeta_to_h5
	from common.gamma_matrices import set_gamma_contract_mode
	# Honour cohsex.in ``gamma_contract_mode`` for the γ̃·γ̃ kernel
	# inside the monolithic pair pipeline.  Mode is module-level (the
	# γ̃ contract sits inside shard_map bodies so threading a kwarg
	# through every call would be churn for no benefit).
	set_gamma_contract_mode(cfg.backend.gamma_contract_mode)

	# ISDF left/right band windows (pair density needs asymmetric ranges)
	band_range_left = (band_slices.b0, band_slices.b3)   # all val + sigma cond
	band_range_right = (band_slices.b1, band_slices.b4)   # sigma val + all cond

	# Chunk sizes (band_chunk / chunk_r / q_chunk / gflat_chunk_size) were
	# picked once by ``plan_gflat_chunks`` in the caller and live in
	# ``chunks``; fit_zeta is a pure consumer.
	mem_est = chunks.get('memory_estimate', {})

	zeta_h5_path = os.path.join(tmp_dir, "zeta_q.h5")
	print_fn(f"\n  Chunked ISDF fitting:")
	print_fn(f"    Band chunks: {chunks['band_chunk']}")
	print_fn(f"    R chunks:    {chunks['chunk_r']} (contiguous r-space)")
	print_fn(f"    Q chunks:    {chunks['q_chunk']}")
	if chunks.get('gflat_chunk_size') is not None:
		print_fn(f"    GFlat cs:    {chunks['gflat_chunk_size']}")
	print_fn(f"    Zeta output: {zeta_h5_path}")

	# Band norms for pseudobands normalization (1.0 for deterministic bands)
	_band_norms = getattr(wfn, 'band_norms', None)

	# IBZ-only writes activate when sym is present, ``LORRAX_FORCE_FULL_BZ``
	# is not set, and the charge centroid set passes orbit closure
	# (checked downstream).  The bispinor V_q orchestrator consumes the
	# IBZ-only ζ̃_C identically to the 2-comp path; see derivation in
	# ``reports/bispinor_ibz_2026-05-16/derivation.md``.
	_write_ibz_only_charge = not bool(int(
		os.environ.get('LORRAX_FORCE_FULL_BZ', '0')))
	# Two cutoffs control the bare-Coulomb / ζ-sphere construction:
	#   * ``bare_coulomb_cutoff_ry`` — V_q's sqrt_v(q+G) mask.
	#   * ``zeta_cutoff_ry``         — the on-disk per-q ζ sphere
	#                                   (writer's ``ngk[q]`` is taken
	#                                   from this).
	# Both default to ``wfn.ecutwfc`` (matches BGW's
	# ``screened_coulomb_cutoff`` default), and both have an upper
	# bound of ``wfn.ecutrho`` (the density grid is the largest cutoff
	# the FFT box can represent).  ``zeta_cutoff_ry`` must be
	# ≥ ``bare_coulomb_cutoff_ry`` — V_q reads ζ̃ at every G inside its
	# bare-Coulomb sphere, so anything the V_q kernel needs has to be
	# stored on disk.
	def _resolve_cutoff(val, label, hi):
		if val is None:
			return float(wfn.ecutwfc)
		v = float(val)
		if v > hi + 1e-9:
			raise ValueError(
				f"{label} = {v} Ry exceeds ecutrho = {hi} Ry "
				f"(the FFT grid can't represent G's past ecutrho).")
		return v

	_ecutrho = float(wfn.ecutrho)
	_zeta_vcoul_cutoff = _resolve_cutoff(
		cfg.head.bare_coulomb_cutoff, "bare_coulomb_cutoff", _ecutrho)
	_zeta_cutoff = _resolve_cutoff(
		cfg.head.zeta_cutoff, "zeta_cutoff", _ecutrho)
	if _zeta_vcoul_cutoff > _zeta_cutoff + 1e-9:
		raise ValueError(
			f"bare_coulomb_cutoff = {_zeta_vcoul_cutoff} Ry > "
			f"zeta_cutoff = {_zeta_cutoff} Ry.  The V_q kernel reads "
			f"ζ̃(q+G) at every G inside its sphere; ζ must be stored at "
			f"least as wide.  Increase zeta_cutoff (≤ ecutrho = "
			f"{_ecutrho} Ry) or lower bare_coulomb_cutoff.")
	print_fn(
		f"    cutoffs: zeta = {_zeta_cutoff:.1f} Ry, "
		f"bare-Coulomb = {_zeta_vcoul_cutoff:.1f} Ry  "
		f"(ecutwfc={float(wfn.ecutwfc):.1f}, ecutrho={_ecutrho:.1f})")
	with timing.section("gw_jax.zeta_fit_chunked"), jax_profile.trace_section("zeta_fit"):
		peak_bytes = fit_zeta_to_h5(
			wfn=wfn, sym=sym, meta=meta,
			centroid_indices=centroid_indices, mesh_xy=mesh_xy,
			chunk_r=chunks['chunk_r'], output_file=zeta_h5_path,
			psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
			band_chunk_size=chunks['band_chunk'],
			q_chunk_size=chunks['q_chunk'],
			bispinor=cfg.bispinor,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			band_norms=_band_norms,
			slab_io_backend=cfg.backend.slab_io,
			gspace_mode=cfg.gspace_mode,
			cusolvermp_charge=cfg.backend.cusolvermp_charge,
			cusolvermp_lu=cfg.backend.cusolvermp_lu,
			gflat_chunk_size=int(chunks.get('gflat_chunk_size', 0)),
			write_ibz_only=_write_ibz_only_charge,
			zeta_cutoff_ry=_zeta_cutoff,
		)

	budget_gb = mem_est.get('budget_gb', cfg.memory.per_device_gb)
	if peak_bytes > 0:
		peak_gb = peak_bytes / 1e9
		# peak_bytes_in_use only tracks the true peak under the BFC allocator.
		# cuda_async (XLA_PYTHON_CLIENT_ALLOCATOR=platform — the FFI default)
		# returns freed transients to its pool, so the reading under-reports;
		# flag it rather than print a misleadingly-low % as if it were faithful.
		_async = (os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform"
		          or os.environ.get("TF_GPU_ALLOCATOR") == "cuda_malloc_async")
		_caveat = ("  [cuda_async under-reports — rerun with "
		           "XLA_PYTHON_CLIENT_ALLOCATOR=default for the true peak]") if _async else ""
		print_fn(f"    GPU high-water mark: {peak_gb:.2f} GB / {budget_gb:.2f} GB budget "
		         f"({100 * peak_gb / budget_gb:.0f}%){_caveat}")

	# Default: no transverse-channel ψ to surface to the caller.
	transverse_wfn_data = None

	# ── Bispinor: fit ζ^{μ_L=1,2,3} on the current-density centroid set ──
	# Same kernel as the charge channel, swapping in the γ̃^i vertex.  Three
	# sequential calls keep peak GPU memory at the scalar-fit level.  Output
	# paths follow the convention zeta_q_mu{1,2,3}.h5 next to zeta_q.h5.
	#
	# Loud-fail guard: if cfg.bispinor=True the transverse ζ fit MUST run,
	# otherwise downstream V_q silently falls back to scalar V_q and then
	# crashes on a full-BZ vs IBZ shape mismatch (ζ_T written by bispinor
	# mode is full-BZ; scalar V_q expects IBZ-only).  See the 2026-05-14
	# CrI3 30 Ry test-bed KNOWN_SANDBOX_ERRORS entry.
	if cfg.bispinor and not getattr(cfg.paths, 'centroids_file_current', None):
		raise ValueError(
			"Bispinor calculation requires centroids_file_current in cohsex.in "
			"(set to the path of a current-density kmeans output, e.g. "
			"centroids_frac_NNN_current.txt from "
			"`centroid.kmeans_cli --density-mode current ...`)."
		)
	if cfg.bispinor and getattr(cfg.paths, 'centroids_file_current', None):
		import dataclasses
		from common.wfn_transforms import load_centroids_band_chunked
		from file_io.centroids import load_centroids

		cents_curr_path = cfg.paths.centroids_file_current
		print_fn(f"\n  [bispinor] fitting ζ^{{μ_L=1,2,3}} on current-density "
		         f"centroids: {cents_curr_path}")
		_, cents_curr_idx, n_rmu_curr = load_centroids(
			cents_curr_path, meta.fft_grid)
		# Round n_rmu_jax to n_proc (legacy field, host-count divisor).
		n_rmu_curr_jax = ((n_rmu_curr + meta.n_proc - 1) // meta.n_proc) * meta.n_proc
		# n_rmu_padded uses world_size (= ∏ p_a over the device mesh).
		# Without this refresh the bispinor transverse fit_zeta inherits
		# the charge-channel padded extent, and the C_q reshape at
		# isdf_fitting.py:1442 trips a TypeError when the transverse
		# centroid count differs from the charge count.
		# ``padded_mu_extent`` also honors the test-only
		# LORRAX_EXTRA_MU_PAD knob (pad-extent-invariance gate).
		from runtime.padding import padded_mu_extent
		_world_size = int(jax.device_count())
		n_rmu_curr_padded = padded_mu_extent(n_rmu_curr, _world_size)
		meta_curr = dataclasses.replace(
			meta,
			n_rmu=int(n_rmu_curr),
			n_rmu_jax=int(n_rmu_curr_jax),
			n_rmu_padded=int(n_rmu_curr_padded),
		)
		# ``sys_dim`` is set dynamically on ``meta`` by gw_jax.main
		# (Meta has no sys_dim field), so dataclasses.replace doesn't
		# carry it over.  Copy it explicitly — fit_zeta_to_h5 reads
		# meta.sys_dim when building the per-q G-flat sphere.
		meta_curr.sys_dim = meta.sys_dim

		with timing.section("gw_jax.load_centroid_wfns_current"):
			psi_curr_rmu_Y, psi_curr_rmuT_X = load_centroids_band_chunked(
				wfn, sym, meta_curr,
				jnp.asarray(cents_curr_idx, dtype=jnp.int32),
				cfg.bispinor, mesh_xy,
				band_range=band_slices.full_range,
				band_chunk_size=chunks['band_chunk'],
			)

		# Per-channel cache hygiene.  The 2026-05-04 bispinor branch needed
		# ``jax.clear_caches()`` here because the original ζ-fit cached
		# functions closed over tracers from the enclosing jit (the
		# UnexpectedTracerError surface).  After the 2026-05-08 open-spin
		# consolidation (commit ce28d50), the cached helpers only capture
		# static config (Mesh, shape, kgrid) — no tracer leaks remain.
		#
		# Keeping the surgical drop on ``_fit_one_rchunk_cache`` (whose
		# cache key includes ``id(psi_G_store)`` and would never hit
		# across channels anyway — drop = memory hygiene, not a workaround).
		# The pair-density caches are intentionally preserved so the three
		# transverse channels share the same n_rmu=n_rmu_current compile.
		import gc
		from isdf import core as _isdf_core

		def _drop_traced_caches():
			_isdf_core._fit_one_rchunk_cache.clear()
			gc.collect()

		for mu_L in (1, 2, 3):
			_drop_traced_caches()
			zeta_mu_path = os.path.join(tmp_dir, f"zeta_q_mu{mu_L}.h5")
			print_fn(f"  [bispinor] μ_L={mu_L} → {zeta_mu_path}")
			with timing.section(f"gw_jax.zeta_fit_chunked_mu{mu_L}"), \
			     jax_profile.trace_section(f"zeta_fit_mu{mu_L}"):
				fit_zeta_to_h5(
					wfn=wfn, sym=sym, meta=meta_curr,
					centroid_indices=jnp.asarray(cents_curr_idx, dtype=jnp.int32),
					mesh_xy=mesh_xy,
					chunk_r=chunks['chunk_r'], output_file=zeta_mu_path,
					psi_rmu_Y=psi_curr_rmu_Y, psi_rmuT_X=psi_curr_rmuT_X,
					band_chunk_size=chunks['band_chunk'],
					q_chunk_size=chunks['q_chunk'],
					bispinor=cfg.bispinor,
					band_range_left=band_range_left,
					band_range_right=band_range_right,
							band_norms=_band_norms,
					slab_io_backend=cfg.backend.slab_io,
					gspace_mode=cfg.gspace_mode,
					cusolvermp_charge=cfg.backend.cusolvermp_charge,
					cusolvermp_lu=cfg.backend.cusolvermp_lu,
					gflat_chunk_size=int(chunks.get('gflat_chunk_size', 0)),
					vertex_mu_L=mu_L,
					# Transverse ζ IBZ-write activates whenever the
					# bispinor V_q orchestrator iterates IBZ q's — same
					# gate the charge ζ uses (LORRAX_FORCE_FULL_BZ off).
					# Orbit-closure of the transverse centroid set is
					# checked downstream in ``fit_zeta_to_h5``; failure
					# is loud per the bispinor IBZ requirement.
					write_ibz_only=(bool(cfg.bispinor)
					                and not bool(int(os.environ.get(
						                'LORRAX_FORCE_FULL_BZ', '0')))),
					zeta_cutoff_ry=_zeta_cutoff,
				)
		# Surface the transverse-centroid ψ to the caller so it can build
		# the second Wfns bundle for σ^B without re-loading from WFN.h5.
		# Keeping these arrays alive across the return is intentional —
		# they're the only way the σ^B kernel can sample ψ at r_{μ_T}.
		transverse_wfn_data = {
			'psi_rmu_Y':       psi_curr_rmu_Y,
			'psi_rmuT_X':      psi_curr_rmuT_X,
			'meta':            meta_curr,
			'centroid_indices': jnp.asarray(cents_curr_idx, dtype=jnp.int32),
		}

	return zeta_h5_path, mem_est, transverse_wfn_data


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None, sym=None, centroid_indices=None):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0) where V_qmunu has shape (nq, μ, μ) (flat-q)
	and G0 is (n_rmu,) ζ_μ(G=0) at q=0.  Downstream consumers that need
	the 3-D-k form reshape inside ``common.fft_helpers.make_flat_k_fft``.

	The legacy ``(1, npol, npol, …)`` leading axes are gone — bispinor
	will introduce a structured ``V_q_bispinor`` NamedTuple (CC, CT, TT)
	rather than packing all polarisation tiles into a uniform tensor,
	because charge and transverse channels use different μ counts.
	"""
	from .compute_vcoul import compute_all_V_q

	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")

	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)

	# V_q memory budget (per rank) — informational only.  The live
	# G-flat V_q path bounds its working set with ``vq_g_chunk_size``
	# (per-q G-chunk) and mesh-sharded ζ slabs; there is no byte-budget
	# chooser to feed any more.  Kept for the log line below.
	if mem_est is None:
		mem_est = {}
	budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory.per_device_gb))
	try:
		from common.gpu_utils import get_device_memory_info
		budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
	except Exception:
		pass

	# Resolved earlier in :func:`fit_zeta` (line ~589) via the shared
	# ``_resolve_cutoff`` helper — defaults to ``wfn.ecutwfc``, max
	# ``wfn.ecutrho``, validated against the ζ-sphere cutoff.  Hoist
	# the resolved value here rather than re-resolving so the two call
	# sites stay in sync (this is the V_q half of the same number
	# zeta_fit wrote into ``isdf_header/zeta_cutoff_ry``).
	if cfg.head.bare_coulomb_cutoff is None:
		vcoul_cutoff_ry = float(wfn.ecutwfc)
	else:
		vcoul_cutoff_ry = float(cfg.head.bare_coulomb_cutoff)
	print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry")
	print_fn(f"    V_q budget:    {budget_gb:.2f} GB")

	from file_io.slab_io import SlabIO

	# ── Bispinor branch ────────────────────────────────────────────────
	# When cfg.bispinor is set AND the 4-channel ζ files were produced by
	# fit_zeta (zeta_q.h5 + zeta_q_mu{1,2,3}.h5), dispatch to the
	# 7-tile orchestrator that streams V^{μ_L,ν_L}_q to a dedicated
	# HDF5 file.  The CC tile (μ_L = ν_L = 0) matches the scalar charge
	# V_q — bit-identically for every sandbox bispinor system, which is
	# sys_dim=2: there the G=0 body is regularised by the 2D truncation
	# (f2d→0) and the mini-BZ head-average is a no-op, so the CC builder's
	# omission of ``v_head_miniBZ`` costs nothing.  (In 3D with
	# mc_average_vcoul_body the two would diverge in one G=0 slot per
	# q≠0 — not currently reachable; see v_q_bispinor CC builder.)  We
	# read the CC tile back as the scalar V_qmunu / G0 the downstream
	# restart_state writer expects.  Σ_X^B / Σ_H^B consumers will read
	# the TT tiles directly via BispinorVqReader.
	zeta_dir = os.path.dirname(zeta_h5_path)
	zeta_T_paths = [
		os.path.join(zeta_dir, f"zeta_q_mu{mu_L}.h5") for mu_L in (1, 2, 3)
	]
	bispinor_ready = (
		cfg.bispinor and all(os.path.exists(p) for p in zeta_T_paths)
	)
	if cfg.bispinor and not bispinor_ready:
		print_fn(
			f"  [bispinor] cfg.bispinor=True but transverse ζ files "
			f"not all present at {zeta_dir}/zeta_q_mu{{1,2,3}}.h5 — "
			f"falling back to scalar V_q.  Did fit_zeta receive "
			f"cfg.centroids_file_current?"
		)

	if bispinor_ready:
		from .v_q_bispinor import (
			BispinorVqReader, tile_dataset_name,
		)
		from file_io.centroids import load_centroids as _load_centroids

		# Reload the transverse centroid indices for the bispinor IBZ
		# cascade.  fit_zeta loaded them earlier but didn't surface them
		# to compute_V_q's signature; reloading is cheap (a text file
		# read) and keeps the bispinor IBZ wiring local to this branch.
		# Orbit-closure of the C/T centroid sets is checked inside
		# ``_resolve_ibz_q_list`` (called per tile by the V_q
		# orchestrator) and silently falls back to full-BZ on failure.
		_use_ibz_bispinor = not bool(int(
			os.environ.get('LORRAX_FORCE_FULL_BZ', '0')))
		if _use_ibz_bispinor:
			_cents_curr_path = cfg.paths.centroids_file_current
			_, _cent_T_idx_np, _ = _load_centroids(
				_cents_curr_path, meta.fft_grid)
			_cent_T_idx_for_orchestrator = np.asarray(
				_cent_T_idx_np, dtype=np.int32)
			_cent_C_idx_for_orchestrator = (
				np.asarray(jax.device_get(centroid_indices),
				           dtype=np.int32)
				if centroid_indices is not None else None)
		else:
			_cent_T_idx_for_orchestrator = None
			_cent_C_idx_for_orchestrator = None

		bispinor_h5_path = os.path.join(zeta_dir, "v_q_bispinor.h5")
		print_fn(f"\n  [bispinor] V_q^{{μ_L,ν_L}} → {bispinor_h5_path}")

		# Charge-channel n_rmu (== meta.n_rmu).  Transverse n_rmu_T comes
		# from the dataset shape on disk — read it from one of the ζ_T
		# files.
		# n_rmu_T from the transverse ζ dataset shape on disk
		# (fit_zeta_to_h5 writes all ζ files in G-flat layout).
		with h5py.File(zeta_T_paths[0], 'r') as f:
			n_rmu_T = int(f['zeta_q_G'].shape[1])
		n_rmu_C = int(meta.n_rmu)

		with timing.section("gw_jax.V_q_compute"), \
		     jax_profile.trace_section("V_q_compute_bispinor"):
			# G-flat path: per-q + G-chunked, one orchestrator per
			# four ζ files.  No legacy compute_V_q_tile chooser /
			# μ × ν tiling / in-V_q FFT — see
			# gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5.
			from .v_q_bispinor import compute_V_q_bispinor_g_flat_to_h5
			from file_io.zeta_reader import ZetaReader
			with ZetaReader(zeta_h5_path, mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zc, \
			     ZetaReader(zeta_T_paths[0], mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zt1, \
			     ZetaReader(zeta_T_paths[1], mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zt2, \
			     ZetaReader(zeta_T_paths[2], mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zt3:
				with mesh_xy:
					compute_V_q_bispinor_g_flat_to_h5(
						zeta_C_loader=zc,
						zeta_T_loaders=(zt1, zt2, zt3),
						output_h5_path=bispinor_h5_path,
						mesh_xy=mesh_xy, kgrid=meta.kgrid,
						fft_grid=meta.fft_grid, bvec=bvec,
						cell_volume=meta.cell_volume,
						sys_dim=meta.sys_dim,
						n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T,
						bare_coulomb_cutoff_ry=vcoul_cutoff_ry,
						bdot=(np.asarray(wfn.bdot, dtype=np.float64)
						       if meta.sys_dim == 0 else None),
						g_chunk=(int(cfg.memory.vq_g_chunk_size)
						         if cfg.memory.vq_g_chunk_size > 0 else None),
						backend=cfg.backend.slab_io,
						print_fn=print_fn,
						sym=sym if _use_ibz_bispinor else None,
						centroid_C_idx=_cent_C_idx_for_orchestrator,
						centroid_T_idx=_cent_T_idx_for_orchestrator,
						use_ibz=_use_ibz_bispinor,
					)

		# Read CC tile + g0 back for downstream restart-state writer.
		# The TT tiles stay on disk; Σ_X^B / Σ_H^B will consume them
		# via BispinorVqReader once those paths land.
		with BispinorVqReader(bispinor_h5_path, mesh_xy,
		                      backend=cfg.backend.slab_io) as reader:
			V_q_raw = reader.get_tile(0, 0)
			G0_all = reader.get_g0_CC()
		# V_q_raw is on disk at LOGICAL n_rmu (the orchestrator strips
		# the V-tile pad before write).  In-memory ψ flows at PADDED
		# n_rmu so the σ_X kernel can broadcast V across G's μ axis.
		# Pad V_q_raw with zeros to match — pad rows of ψ are zero
		# (Phase 3a invariant), so zero-padding V is exact.
		if int(V_q_raw.shape[-1]) < int(meta.n_rmu_padded):
			pad = int(meta.n_rmu_padded) - int(V_q_raw.shape[-1])
			V_q_raw = jnp.pad(V_q_raw, ((0, 0), (0, pad), (0, pad)))
		if G0_all is not None and int(G0_all.shape[-1]) < int(meta.n_rmu_padded):
			G0_all = jnp.pad(G0_all,
			                 ((0, 0), (0, int(meta.n_rmu_padded) - int(G0_all.shape[-1]))))
	else:
		# Scalar (non-bispinor) path.  ``compute_all_V_q`` dispatches on
		# the on-disk ζ layout: G-flat (the only thing fit_zeta writes)
		# routes to ``v_q_g_flat.compute_all_V_q_g_flat``; any other
		# layout raises.  ``ZetaReader`` is the V_q reader of record —
		# it serves the writer's per-q WFN.h5-style G-sphere directly.
		from file_io.zeta_reader import ZetaReader
		with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
			with ZetaReader(zeta_h5_path, mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zeta_io:
				with mesh_xy:
					V_q_raw, G0_all = compute_all_V_q(
						zeta_io,
						kgrid=meta.kgrid, fft_grid=meta.fft_grid,
						bvec=bvec, cell_volume=meta.cell_volume,
						mesh_xy=mesh_xy,
						sys_dim=meta.sys_dim,
						bdot=np.asarray(wfn.bdot, dtype=np.float64)
							if meta.sys_dim == 0 else None,
						mc_average_vcoul_body=cfg.head.mc_average_vcoul_body,
						bare_coulomb_cutoff=vcoul_cutoff_ry,
						bgw_v_grid_fn=bgw_v_grid_fn,
						sym=sym,
						centroid_indices=(
							np.asarray(jax.device_get(centroid_indices),
							           dtype=np.int32)
							if centroid_indices is not None else None),
						g_chunk_size=int(cfg.memory.vq_g_chunk_size),
					)

	# Write G0 = ζ_μ(G=0) at q=0 back to zeta file via SlabIO's deferred
	# attr path (small; rank-0-only after MPI-IO file is closed).
	from file_io._slab_io_allgather import _to_host as _gather_to_host
	G0_gathered = _gather_to_host(G0_all)
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			if 'g0_mu' in f:
				del f['g0_mu']
			f.create_dataset('g0_mu', data=np.asarray(G0_gathered))
	jax.experimental.multihost_utils.sync_global_devices("g0_write")

	# Scalar V_qmunu is just (nq, μ, μ).  The (1, npol, npol) leading
	# axes of the legacy 8-D layout were never used in scalar mode and
	# have no place once bispinor switches to a structured tile container
	# (CC + CT(3) + TT(3,3) NamedTuple) since the μ counts differ across
	# polarisation tiles.  See agent/v_q_perf design discussion 2026-05-08.
	V_qmunu = V_q_raw

	G0 = G0_gathered
	while G0.ndim > 1:
		G0 = G0[0]

	print_fn(f"\n  V_q computed:")
	print_fn(f"    Shape: {V_qmunu.shape}")
	# V_q_raw is now flat-q (nq, μ, μ); q=0 slab is V_q_raw[0].
	print_fn(f"    V_q=0 trace: {jnp.trace(V_q_raw[0]).real:.4f}")
	return V_qmunu, G0


def build_wavefunction_bundle(
	wfn, sym, meta, band_slices, mesh_xy,
	*, psi_rmu_Y, psi_rmuT_X, enk_full=None, print_fn=print,
):
	"""Build 4-copy Wavefunctions bundle from the two centroid-sampled
	arrays produced by ``load_centroids_band_chunked``.
	"""
	from .wavefunction_bundle import build_wavefunctions
	from common.wfn_transforms import get_enk_bandrange

	if enk_full is None:
		enk_full, _ = get_enk_bandrange(
			wfn, sym, band_slices.full_range,
			(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

	wfns = build_wavefunctions(
		psi_rmu_Y, psi_rmuT_X,
		enk_full=enk_full, slices=band_slices, mesh_xy=mesh_xy)

	print_fn(f"  Wavefunctions built (b0:b4={band_slices.nb_full} bands, "
	         f"4 sharded copies: xn/xr/yr/yn)")
	return wfns


def prepare_isdf_and_wavefunctions(
	*, cfg, wfn, sym, meta, centroid_indices, band_slices,
	mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None, **_ignored,
):
	"""ISDF pipeline (non-restart path reads top-to-bottom):

	  1. ``compute_optimal_chunks`` → chunk plan (band/r/q chunk sizes).
	  2. ``load_centroids_band_chunked`` → ψ at centroids for [b0, b4).
	  3. ``fit_zeta`` → ζ.h5 (consumes ψ slices for pair density).
	  4. ``compute_V_q`` → V_qmunu, G0 (reads ζ from disk).
	  5. Flush V_q / G0 / enk + W0 placeholder to restart H5 (mode="w").
	  6. ``build_wavefunctions`` → 4-copy Wavefunctions bundle (reuses ψ).
	  7. Append ``psi_full_y`` (= wfns.psi_yr) to restart H5 (mode="a").

	Returns SimpleNamespace(V_qmunu, wf_bundle).
	"""
	from file_io import write_restart_state_to_h5, save_restart_state_per_proc
	from common.wfn_transforms import load_centroids_band_chunked

	if not cfg.restart:
		from common.wfn_transforms import get_enk_bandrange

		with mesh_xy:
			# Plan chunk sizes ONCE — the single production planner owns
			# band_chunk / chunk_r / q_chunk / gflat_chunk_size, the rank
			# floor P_min, and the binding-stage report.
			from gw.gflat_memory_model import plan_gflat_chunks
			mem = cfg.memory
			nb_total = ((band_slices.b3 - band_slices.b0)
			            + (band_slices.b4 - band_slices.b1))
			# Q-axis on disk: conservative full-BZ (the transverse path
			# writes IBZ-only, which is smaller).
			_ngkmax = int(getattr(meta, 'ngkmax', 0)) or int(0.06 * meta.n_rtot)
			gflat_plan = plan_gflat_chunks(
				meta=meta, mesh_xy=mesh_xy,
				nb_total=nb_total, ngkmax=_ngkmax,
				n_q_disk=int(meta.nk_tot),
				budget_gb=float(mem.per_device_gb),
				target_utilization=(mem.chunk_target_utilization
				                    if mem.chunk_target_utilization > 0 else None),
				is_bispinor=bool(cfg.bispinor),
				max_chunks=64,
				r_chunk_override=(int(mem.r_chunk_override)
				                  if mem.r_chunk_override > 0 else None),
				band_chunk_override=(int(mem.band_chunk_size)
				                     if mem.band_chunk_size > 0 else None),
				gflat_chunk_size_override=(int(mem.gflat_chunk_size)
				                           if mem.gflat_chunk_size > 0 else None),
			)
			if jax.process_index() == 0:
				print0("")
				print0(gflat_plan.format())
			if gflat_plan.p_min > mesh_xy.devices.size:
				print0(f"  [planner] WARNING: rank floor P_min="
				       f"{gflat_plan.p_min} exceeds the {mesh_xy.devices.size} "
				       f"ranks in this mesh; the persistent ÷P floor will not "
				       f"fit the budget (expect OOM).  Add ranks or raise "
				       f"memory_per_device_gb.")
			chunks = {
				'band_chunk': int(gflat_plan.band_chunk),
				'chunk_r': int(gflat_plan.r_chunk),
				'q_chunk': int(gflat_plan.q_chunk),
				'gflat_chunk_size': int(gflat_plan.gflat_chunk_size),
				'gflat_hwm_gb': gflat_plan.hwm_bytes / 1e9,
				'memory_estimate': {
					'peak_estimate_gb': gflat_plan.hwm_bytes / 1e9,
					'budget_gb': float(mem.per_device_gb),
					'bottleneck': gflat_plan.bottleneck,
					'available_vcoul_gb': max(
						0.0, gflat_plan.budget_bytes
						- gflat_plan.persistent_bytes) / 1e9,
				},
			}

			# Load centroid ψ once for the full [b0, b4) range; reused by
			# both the zeta fit (sliced into halves internally) and the
			# downstream Wavefunctions bundle.
			with timing.section("gw_jax.load_centroid_wfns"):
				psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
					wfn, sym, meta, centroid_indices, cfg.bispinor, mesh_xy,
					band_range=band_slices.full_range,
					band_chunk_size=chunks['band_chunk'],
				)

			zeta_path, mem_est, transverse_wfn_data = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, band_slices, tmp_dir,
				psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print0)
			# Profiling helper: LORRAX_EXIT_AFTER_ZETA=1 short-circuits
			# the pipeline right after ζ-fit, before the expensive V_q
			# stage.  Combine with LORRAX_MAX_RCHUNKS=N + LORRAX_RCHUNK_DEBUG=1
			# for fast per-r-chunk timing sweeps.
			if os.environ.get("LORRAX_EXIT_AFTER_ZETA"):
				if jax.process_index() == 0:
					print("[profile] LORRAX_EXIT_AFTER_ZETA set: "
					      "exiting cleanly after fit_zeta.", flush=True)
				raise SystemExit(0)
			# P4 — pre-V_q.  Whatever's still in HBM after fit_zeta
			# returns forms the persistent baseline that V_q's transient
			# peak stacks on top of.  Same env gate as the ζ-fit probes
			# (LORRAX_MEM_DEBUG=1).  Round-1 addition.
			from gw.isdf_fitting import mem_probe as _mem_probe
			_mem_probe("pre_v_q")
			V_qmunu, G0 = compute_V_q(
				zeta_path, wfn, meta, mesh_xy, cfg,
				mem_est=mem_est, print_fn=print0,
				bgw_v_grid_fn=bgw_v_grid_fn,
				sym=sym, centroid_indices=centroid_indices)
			# P5 — post-V_q.  V_q's transient peak just happened inside
			# compute_V_q; this probe captures what survives (V_qmunu,
			# G0) plus anything held over from ζ-fit.  Combined with P4
			# and the V_q HLO buffer-assignment.txt this lets us model
			# V_q's contribution to overall HBM peak.  Round-1 addition.
			if os.environ.get("LORRAX_MEM_DEBUG"):
				jax.block_until_ready(V_qmunu)
			_mem_probe("post_v_q")

			enk_full, _ = get_enk_bandrange(
				wfn, sym, band_slices.full_range,
				(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

			# Flush V_q / G0 / enk + W0 placeholder immediately.  Pass
			# kgrid so BSE downstream can recover the (nkx, nky, nkz)
			# split from flat-q V_qmunu without re-reading the WFN.
			write_restart_state_to_h5(
				tensors_filename,
				V_qmunu=V_qmunu, G0_mu_nu=G0, enk_full=enk_full,
				init_W0=True, mesh=mesh_xy, backend=cfg.backend.slab_io,
				mode="w", kgrid=tuple(int(v) for v in meta.kgrid),
			)

			with timing.section("gw_jax.wavefunction_setup"):
				wfns = build_wavefunction_bundle(
					wfn, sym, meta, band_slices, mesh_xy,
					psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
					enk_full=enk_full, print_fn=print0)

				# Bispinor: build a second Wfns bundle on the
				# transverse centroid set so Σ^B can sample ψ at
				# r_{μ_T} without re-reading WFN.h5.
				wfns_transverse = None
				if transverse_wfn_data is not None:
					wfns_transverse = build_wavefunction_bundle(
						wfn, sym,
						transverse_wfn_data['meta'],
						band_slices, mesh_xy,
						psi_rmu_Y=transverse_wfn_data['psi_rmu_Y'],
						psi_rmuT_X=transverse_wfn_data['psi_rmuT_X'],
						enk_full=enk_full, print_fn=print0)
					print0(f"  [bispinor] σ^B-side Wfns built on "
					       f"n_rmu_T={transverse_wfn_data['meta'].n_rmu} "
					       f"transverse centroids")

			# Append ψ to the now-open restart file.
			write_restart_state_to_h5(
				tensors_filename,
				psi_full_y=wfns.psi_yr, mesh=mesh_xy,
				backend=cfg.backend.slab_io, mode="a",
			)
		save_restart_state_per_proc(
			os.path.join(tmp_dir, "isdf_tensors"),
			V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices)
			V_qmunu = rs.V_qmunu
			print0("  Loaded restart tensors from H5.")
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_rmu_Y=rs.psi_rmu_Y, psi_rmuT_X=rs.psi_rmuT_X,
				enk_full=rs.enk_full, print_fn=print0)
			# Restart path doesn't yet round-trip the transverse
			# Wfns through the restart file; bispinor restart will
			# need a second psi_full_y dataset (per-channel).  Mark
			# as not-yet-supported so consumers fail loud.
			wfns_transverse = None

	return SimpleNamespace(
		V_qmunu=V_qmunu,
		wf_bundle=wfns,
		wf_bundle_transverse=wfns_transverse,
	)

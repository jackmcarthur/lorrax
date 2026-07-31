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
# Named-barrier helper: annotates failures with the barrier name and
# no-ops at P=1; also avoids relying on ``jax.experimental.multihost_utils``
# being attribute-reachable off a bare ``import jax`` (an import-order
# accident).  (audit fix/zq 2026-07-28)
from common.collectives import barrier



# Backward-compatible re-exports
from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401

# Canonical env grammar for this layer.  ``gw_config`` is deliberately
# jax-free, so importing it here adds nothing to the import graph that
# ``read_lorrax_input`` above did not already add.  See the module comment
# in gw_config for why this vocabulary is duplicated rather than imported
# from ``isdf.core`` (which imports jax) — and for the drift gate that
# keeps the copies identical.
from .gw_config import (
	env_bool,
	active_zeta_truncating_knobs,
	classify_xla_pool,
	resolve_xla_gpu_memory_env,
)


def _check_zeta_h5_matches_basis(zeta_h5_path, n_rmu, print_fn=print,
                                 *, fft_grid=None):
	"""Fail fast when ``tmp/zeta_q.h5`` belongs to a different ISDF basis.

	``tmp_dir`` is hardwired to ``<input_dir>/tmp`` and the ζ file is a fixed
	``zeta_q.h5`` -- unlike its sibling ``isdf_tensors_{n_rmu}.h5``, it is NOT
	namespaced by centroid count.  So two runs launched from the same directory
	with different ``centroids_file`` share one ζ file.  The writer opens it
	``mode='a'`` and only discovers the clash deep inside the slab write, as

	  ValueError: could not broadcast input array from shape (144,276,8603)
	                                          into shape (144,1194,8603)

	i.e. *after* the entire multi-hour ζ fit has already been computed and
	thrown away.  Check the precondition up front instead: it costs one HDF5
	header read and turns an hour of wasted compute into an actionable message.

	Three invariants, all off one header read:

	1. **μ extent.**  Historically this probed ``f['zeta_q']`` only — the
	   *legacy r-space* dataset name.  Every production run since the G-flat
	   migration writes ``zeta_q_G`` instead, so ``f.get('zeta_q')`` returned
	   ``None`` and the guard silently passed on exactly the files it was
	   written to protect.  Both names are probed now.
	2. **``zeta_is_done``.**  The writer stamps this ``False`` up front and
	   flips it via ``mark_zeta_done`` only after the last chunk lands.  It
	   was written but never read by anybody, so a ζ from a job that died
	   mid-write (the campaign produced several) was indistinguishable on
	   disk from a complete one.
	3. **Centroid grid.**  Two centroid sets of the *same* size on different
	   FFT grids pass (1) and produce silently wrong pair densities.
	"""
	if not os.path.exists(zeta_h5_path):
		return
	existing = None
	zeta_done = None
	header_grid = None
	try:
		with h5py.File(zeta_h5_path, 'r') as f:
			# G-flat (production): (n_q, n_rmu, ngkmax).
			# r-space (legacy):    (n_q, n_rtot, n_rmu).
			for _name, _mu_axis in (('zeta_q_G', 1), ('zeta_q', 2)):
				dset = f.get(_name)
				if dset is not None and dset.ndim == 3:
					existing = int(dset.shape[_mu_axis])
					break
			hdr = f.get('isdf_header')
			if hdr is not None:
				if 'zeta_is_done' in hdr:
					zeta_done = bool(np.asarray(hdr['zeta_is_done'])[()])
				cent = hdr.get('centroids/r_mu_fft_idx')
				if cent is not None:
					header_grid = np.asarray(cent, dtype=np.int64)
	except Exception as exc:
		# Unreadable/partial file: say so, then let the writer deal with it.
		print_fn(f"  [zeta guard] could not read {zeta_h5_path} "
		         f"({type(exc).__name__}: {exc}); continuing.")
		return
	if existing is not None and existing != int(n_rmu):
		raise ValueError(
			f"{zeta_h5_path} holds a ζ for n_mu={existing}, but this run has "
			f"n_mu={n_rmu}.  tmp/ is derived from the input file's directory "
			f"and zeta_q.h5 is not namespaced by centroid count, so two runs "
			f"started from the same directory with different centroids_file "
			f"values collide here.  Give each run its own directory (put its "
			f"input file there), or move/delete the stale ζ.  Refusing now "
			f"rather than after the fit.")
	if zeta_done is False:
		print_fn(
			f"  *** LORRAX SANITY: {zeta_h5_path} has zeta_is_done=False — "
			f"it was left behind by a ζ-fit that did NOT finish writing.  "
			f"This run will overwrite it, which is fine; but if you meant to "
			f"REUSE it, do not: its trailing q-blocks are undefined. ***")
	if header_grid is not None and header_grid.shape[0] != int(n_rmu):
		raise ValueError(
			f"{zeta_h5_path} isdf_header lists {header_grid.shape[0]} "
			f"centroids but this run has n_mu={n_rmu} — the header and the ζ "
			f"dataset in that file disagree, so it is corrupt.  Delete it.")
	if (header_grid is not None and fft_grid is not None
			and header_grid.size):
		fg = np.asarray(fft_grid, dtype=np.int64).reshape(3)
		# Centroid FFT indices are bounded by the grid they were kmeans'd
		# on; an index at or beyond this run's grid proves a different grid.
		if bool(np.any(header_grid.max(axis=0) >= fg)):
			raise ValueError(
				f"{zeta_h5_path} was built on a different FFT grid: its "
				f"centroid indices reach {tuple(header_grid.max(axis=0))} "
				f"but this run's grid is {tuple(int(v) for v in fg)}.  The "
				f"pair densities would be sampled at the wrong points.  "
				f"Delete the stale ζ or use a separate run directory.")


#: Bump when the provenance schema changes meaning (forces a refit of
#: every ζ stamped by an older LORRAX, rather than a false match).
_ZETA_PROVENANCE_SCHEMA = 1


def _zeta_fit_provenance(*, wfn, meta, cfg, band_range_left, band_range_right,
                         zeta_cutoff, zeta_vcoul_cutoff, write_ibz_only,
                         band_norms):
	"""Canonical JSON description of everything the ζ fit consumed.

	Every entry is an input that CHANGES ζ numerically.  Deliberately
	EXCLUDED: anything device-count dependent (``n_rmu_padded``, the chunk
	plan, the mesh shape) — ζ is device-count invariant by design (the
	2026-07 z_q invariance work), so a P=4 ζ is reusable at P=80 and
	including P would defeat the whole feature.

	Also excluded: the centroid table itself.  It is already on disk in
	full (``isdf_header/centroids/r_mu_fft_idx``) and is compared
	element-wise by :func:`_zeta_reuse_ok` — a hash here would only add a
	second, weaker check.

	``wfn_bytes`` identifies the source WFN.h5 without reading it.  mtime
	is deliberately NOT included: copying/restoring a WFN.h5 changes mtime
	without changing content, and a spurious multi-hour refit is a worse
	outcome than the (contrived) same-path-same-size-different-content
	case.

	``zeta_rcond`` / ``zeta_ridge`` record the EFFECTIVE values, i.e. after
	``LORRAX_ZETA_RCOND`` / ``LORRAX_ZETA_RIDGE`` are applied exactly as
	``isdf/core._replicated_chol`` applies them.  Recording the cfg values
	instead would be a correctness hole: a rerun that drops the env
	override would match the provenance and silently reuse a ζ fit at a
	different conditioning cutoff.

	``write_ibz_only`` is the REQUESTED value.  ``fit_zeta_to_h5`` may
	flip it off when the orbit-closure check fails — but that check is a
	deterministic function of (centroids, symmetry), both of which are
	pinned by other entries here, so the effective value is pinned too.
	"""
	import json
	from isdf.core import deprecated_env_record as _dep_env_record
	# ``wfn._filename`` is the same source path fit_zeta_to_h5 copies
	# mf_header from — the authoritative identity of the ζ's input WFN.
	wfn_path = getattr(wfn, '_filename', None) or ''
	try:
		wfn_bytes = int(os.path.getsize(wfn_path)) if wfn_path else -1
	except OSError:
		wfn_bytes = -1
	bn = None
	if band_norms is not None:
		arr = np.asarray(band_norms, dtype=np.float64)
		bn = [int(arr.size), float(arr.sum()), float(arr.max()) if arr.size else 0.0]
	prov = {
		'schema':               _ZETA_PROVENANCE_SCHEMA,
		'n_rmu':                int(meta.n_rmu),
		'band_range_left':      [int(band_range_left[0]), int(band_range_left[1])],
		'band_range_right':     [int(band_range_right[0]), int(band_range_right[1])],
		'bispinor':             bool(cfg.bispinor),
		'gspace_mode':          str(cfg.gspace_mode),
		'zeta_cutoff_ry':       round(float(zeta_cutoff), 9),
		'bare_coulomb_cutoff':  round(float(zeta_vcoul_cutoff), 9),
		# EFFECTIVE (env-overridden) values, recorded via the SAME
		# non-empty-env-wins rule the factor sites apply
		# (isdf/core.deprecated_env_record → _env_override_raw — ONE
		# implementation, no inline mirror to drift; audit fix/zq
		# 2026-07-28): the env forms are DEPRECATED (scorecard AV) but
		# still win when non-empty, so the provenance must keep recording
		# what the fit actually used.  The recorded string is
		# byte-identical to the historical format in every case that
		# ever produced a reusable ζ.
		'zeta_ridge':           _dep_env_record(
			"LORRAX_ZETA_RIDGE", cfg.backend.zeta_ridge),
		'zeta_rcond':           _dep_env_record(
			"LORRAX_ZETA_RCOND", cfg.backend.zeta_rcond),
		'charge_zeta_solve':    str(cfg.backend.charge_zeta_solve),
		'gamma_contract_mode':  str(cfg.backend.gamma_contract_mode),
		'write_ibz_only':       bool(write_ibz_only),
		'vertex_mu_L':          0,
		'band_norms':           bn,
		'fft_grid':             [int(x) for x in np.asarray(meta.fft_grid).reshape(3)],
		'ecutwfc':              round(float(wfn.ecutwfc), 9),
		'ecutrho':              round(float(wfn.ecutrho), 9),
		'wfn_file':             os.path.abspath(wfn_path) if wfn_path else '',
		'wfn_bytes':            wfn_bytes,
	}
	return json.dumps(prov, sort_keys=True)


def _zeta_reuse_ok(zeta_h5_path, provenance_json, centroid_fft_idx,
                   print_fn=print):
	"""Can we skip the ζ fit and reuse ``zeta_h5_path`` as-is?

	Returns ``True`` only when EVERY one of these holds:

	  1. ``LORRAX_FORCE_REFIT`` is not set to a truthy value;
	  2. the file exists and its ``isdf_header`` is readable;
	  3. ``zeta_is_done`` is True (the writer flipped it after the last
	     H5Dwrite drained — a crashed fit leaves it False);
	  4. ``fit_provenance`` is present AND byte-identical to this run's;
	  5. the on-disk centroid table equals this run's centroid indices.

	Anything unexpected — missing attr, unreadable file, legacy header —
	returns False, i.e. REFIT.  Every failure mode costs compute; none
	costs correctness.  That asymmetry is the whole design: this cache
	sits in front of a multi-hour step whose silent misuse produced a
	−135 eV QP gap once already (job 7874375, the restart-window bug).
	"""
	# Canonical boolean grammar (isdf/core._env_bool — same set as
	# _slab_io_mpi_host._env_flag): under the old hand-rolled
	# ``not in ('', '0', 'false', 'False')`` parse, ``=off``/``=no``
	# counted as truthy and forced a multi-hour refit (audit fix/zq
	# 2026-07-28).
	from isdf.core import _env_bool
	if _env_bool('LORRAX_FORCE_REFIT', False):
		print_fn("    [zeta reuse] LORRAX_FORCE_REFIT set — refitting.")
		return False
	if not os.path.exists(zeta_h5_path):
		return False
	try:
		from file_io.isdf_header import read_isdf_header
		hdr = read_isdf_header(zeta_h5_path)
	except Exception as exc:
		print_fn(f"    [zeta reuse] {zeta_h5_path}: unreadable isdf_header "
		         f"({exc}) — refitting.")
		return False
	if not bool(hdr.zeta_is_done):
		print_fn(f"    [zeta reuse] {zeta_h5_path} is INCOMPLETE "
		         f"(zeta_is_done=False, i.e. a previous fit died mid-write) "
		         f"— refitting.")
		return False
	if hdr.fit_provenance is None:
		print_fn(f"    [zeta reuse] {zeta_h5_path} carries no fit_provenance "
		         f"(written by an older LORRAX) — cannot verify what it was "
		         f"fit for, so refitting.")
		return False
	if str(hdr.fit_provenance) != str(provenance_json):
		import json
		try:
			old = json.loads(hdr.fit_provenance)
			new = json.loads(provenance_json)
			diff = sorted(k for k in set(old) | set(new)
			              if old.get(k) != new.get(k))
			detail = "; ".join(
				f"{k}: on-disk={old.get(k)!r} now={new.get(k)!r}" for k in diff)
		except Exception:
			detail = "(provenance unparseable)"
		print_fn(f"    [zeta reuse] {zeta_h5_path} was fit under DIFFERENT "
		         f"inputs — refitting.  Changed: {detail}")
		return False
	try:
		want = np.asarray(centroid_fft_idx, dtype=np.int32)
		got = np.asarray(hdr.r_mu_fft_idx, dtype=np.int32)
	except Exception:
		return False
	if want.shape != got.shape or not np.array_equal(want, got):
		print_fn(f"    [zeta reuse] {zeta_h5_path} holds a DIFFERENT centroid "
		         f"set ({got.shape} vs {want.shape}) — refitting.")
		return False
	return True


def _centroid_table_md5(centroid_fft_idx) -> str:
	"""Canonical content hash of a centroid fft-index table: int64,
	C-order bytes → md5 hexdigest.

	Stamped on the restart tensors file at write and verified at restart
	load, so a restart proves the σ quadrature BASIS — not just its size —
	matches this run's centroid file(s).  A regenerated centroid file with
	the SAME count but different points (kmeans reruns plausibly produce
	this) used to pass the count-only guard and evaluate Σ with ψ sampled
	at the wrong r_μ — silently wrong physics (pattern #10).  The sibling
	ζ-reuse cache compares its centroid table element-wise
	(:func:`_zeta_reuse_ok`); this is the same check compressed to an HDF5
	attr.  (audit fix/zq 2026-07-28)
	"""
	import hashlib
	arr = np.ascontiguousarray(
		np.asarray(jax.device_get(centroid_fft_idx), dtype=np.int64))
	return hashlib.md5(arr.tobytes()).hexdigest()


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
	_check_zeta_h5_matches_basis(zeta_h5_path, int(meta.n_rmu), print_fn,
	                             fft_grid=meta.fft_grid)
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
	#
	# ``env_bool``, not ``bool(int(os.environ.get(...)))``: the latter
	# accepts only decimal digits, so the natural spellings of this knob —
	# ``=1`` works but ``=true``/``=on``/``=yes`` raise a bare
	# ``ValueError: invalid literal for int() with base 10`` from inside
	# ISDF setup, and ``=2`` silently means "on".  All five sites that read
	# this variable (three here, ``gw/v_q_g_flat.py``, ``gw/screening.py``)
	# were converted together — a knob with two grammars is worse than one
	# with a single wrong grammar, because then the failure depends on
	# which code path reads it first.
	_write_ibz_only_charge = not env_bool(
		'LORRAX_FORCE_FULL_BZ', False, print_fn=print_fn)
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
	# ── Bispinor PRE-FLIGHT (audit fix/zq 2026-07-28): resolve the
	# transverse distributed-LU contract BEFORE any compute.  The
	# divisibility refusal (isdf/core._resolve_solver_kind_transverse)
	# used to fire only inside the transverse fit_zeta_to_h5 calls below,
	# which run AFTER the charge fit completes — and a bispinor run always
	# refits the charge channel (ζ reuse is charge-only), so an explicit
	# distributed_lu whose transverse centroid count does not divide the
	# mesh burned the multi-hour charge fit before the inevitable
	# ValueError.  Every input needed to refuse — mesh shape,
	# distributed_lu, the transverse centroid count — is known right here
	# (one text-file read), so a doomed run refuses in seconds; the same
	# call announces an ``auto`` demotion up front (rank 0).  The
	# per-channel resolution inside fit_zeta_to_h5 later re-resolves
	# identically (defense in depth).
	if cfg.bispinor:
		# Requirement check hoisted with the pre-flight (it used to sit
		# after the charge fit, same late-refusal shape).
		if not getattr(cfg.paths, 'centroids_file_current', None):
			raise ValueError(
				"Bispinor calculation requires centroids_file_current in cohsex.in "
				"(set to the path of a current-density kmeans output, e.g. "
				"centroids_frac_NNN_current.txt from "
				"`centroid.kmeans_cli --density-mode current ...`)."
			)
		from file_io.centroids import load_centroids as _load_cent_pf
		from isdf.core import _resolve_solver_kind_transverse
		_, _, _n_rmu_T_pf = _load_cent_pf(
			cfg.paths.centroids_file_current, meta.fft_grid)
		_resolve_solver_kind_transverse(
			mesh_xy, cfg.backend.distributed_lu,
			n_rmu_logical=int(_n_rmu_T_pf))
	# ── ζ REUSE: skip the fit when tmp/zeta_q.h5 is complete AND provably
	# the same fit.  Before this, a rerun in the same directory always
	# refit (gw_init only VALIDATED the μ extent), costing 20+ min at
	# fixture scale and ~22 min at MoS2 12×12/P=80 for a byte-identical
	# result.  Reuse is charge-channel-only: the bispinor branch below
	# also produces ``transverse_wfn_data``, which is not on disk, so a
	# bispinor run always refits.  Override with LORRAX_FORCE_REFIT=1.
	_provenance = _zeta_fit_provenance(
		wfn=wfn, meta=meta, cfg=cfg,
		band_range_left=band_range_left, band_range_right=band_range_right,
		zeta_cutoff=_zeta_cutoff, zeta_vcoul_cutoff=_zeta_vcoul_cutoff,
		write_ibz_only=_write_ibz_only_charge, band_norms=_band_norms)
	_reuse = (not cfg.bispinor) and _zeta_reuse_ok(
		zeta_h5_path, _provenance, centroid_indices, print_fn=print_fn)
	if _reuse:
		print_fn("")
		print_fn("  " + "=" * 68)
		print_fn(f"  REUSING the existing ζ at {zeta_h5_path} — FIT SKIPPED.")
		print_fn( "  isdf_header/zeta_is_done is True, the centroid table")
		print_fn( "  matches, and fit_provenance is identical to this run's")
		print_fn( "  inputs (band windows, cutoffs, solver knobs, source WFN).")
		print_fn( "  Set LORRAX_FORCE_REFIT=1 to refit unconditionally.")
		print_fn("  " + "=" * 68)
		print_fn("")
		return zeta_h5_path, mem_est, None

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
			distributed_cholesky=cfg.backend.distributed_cholesky,
			distributed_lu=cfg.backend.distributed_lu,
			zeta_ridge=cfg.backend.zeta_ridge,
			charge_zeta_solve=cfg.backend.charge_zeta_solve,
			distributed_zeta_solve=cfg.backend.distributed_zeta_solve,
			zeta_rcond=cfg.backend.zeta_rcond,
			gflat_chunk_size=int(chunks.get('gflat_chunk_size', 0)),
			write_ibz_only=_write_ibz_only_charge,
			zeta_cutoff_ry=_zeta_cutoff,
		)

	# Stamp what this ζ was fit FOR, so a later run can reuse it.  AFTER
	# the fit (and therefore after ``mark_zeta_done`` inside
	# fit_zeta_to_h5) on purpose: a job killed between the two leaves a
	# complete-but-unstamped file, which _zeta_reuse_ok refits.  Rank 0
	# only, then a barrier so no rank races ahead of the write.
	#
	# EXCEPT when a truncating knob was in force.  ``LORRAX_MAX_RCHUNKS=N``
	# breaks the r-chunk loop after N chunks (gw/isdf_fitting.py) and the
	# writer downstream of the loop still calls ``mark_zeta_done``, so the
	# partial ζ is stamped COMPLETE on disk.  Provenance records the
	# CONFIGURATION, which a later production run in the same directory
	# reproduces exactly — so stamping here would make _zeta_reuse_ok
	# reuse a truncated ζ and produce silently wrong physics from a
	# profiling knob.  Refusing the stamp breaks that chain outright
	# (rule 4 of _zeta_reuse_ok: no provenance ⇒ refit).
	#
	# The writer now consults the SAME knob list before calling
	# ``mark_zeta_done``, so a truncated file also carries
	# ``zeta_is_done=False``.  The two guards stay separate on purpose —
	# provenance answers "may a later run REUSE this", zeta_is_done answers
	# "did the writer FINISH" — and either alone stops the reuse.
	_trunc = active_zeta_truncating_knobs()
	if _trunc:
		_names = ", ".join(f"{k}={v}" for k, v in _trunc)
		print_fn("")
		print_fn("  " + "!" * 68)
		print_fn(f"  *** LORRAX SANITY: {_names} truncated this ζ fit. ***")
		print_fn(f"  {zeta_h5_path} is INCOMPLETE and is NOT being stamped")
		print_fn( "  with fit_provenance, so no later run can reuse it.  The")
		print_fn( "  writer also left isdf_header/zeta_is_done False, so no")
		print_fn( "  restart path will trust it either.  Profiling only —")
		print_fn( "  delete this file before any production run from this")
		print_fn( "  directory.")
		print_fn("  " + "!" * 68)
		print_fn("")
	elif jax.process_index() == 0:
		try:
			from file_io.isdf_header import stamp_fit_provenance
			stamp_fit_provenance(zeta_h5_path, _provenance)
		except Exception as exc:
			# Non-fatal: the ζ itself is fine, it just won't be reusable.
			print_fn(f"    [zeta provenance] not stamped ({exc}); this ζ "
			         f"will be refit on the next run.")
	barrier("zeta_provenance")

	budget_gb = mem_est.get('budget_gb', cfg.memory.per_device_gb)
	if peak_bytes > 0:
		peak_gb = peak_bytes / 1e9
		# WHERE DID THIS NUMBER COME FROM?  ``fit_zeta_to_h5._track_peak``
		# prefers ``memory_stats()['peak_bytes_in_use']`` and falls back to
		# an nvidia-smi whole-GPU sample when that is 0/absent — and the
		# caller cannot tell which fired.  Under the ``platform`` allocator
		# the arena reports bytes_limit=0 AND peak_bytes_in_use=0
		# (measured, job 7882447), so every figure printed there is the
		# nvidia-smi fallback: the whole card, other processes included.
		#
		# Four bugs in the previous three lines, all silent:
		#   * ``== "platform"`` was case-SENSITIVE while jax lowercases
		#     (jaxlib/xla_client.py:190), so ``=PLATFORM`` printed bare;
		#   * it never matched ``cuda_async``, which is what
		#     ``config/frontera/ffi_env.sh:24`` deploys;
		#   * it called ``platform`` "cuda_async" — three distinct
		#     allocators, and ``platform`` is plain cudaMalloc;
		#   * its ``TF_GPU_ALLOCATOR`` clause was dead (inert for jax).
		# And its premise — "cuda_async under-reports" — was not
		# reproduced: peak_bytes_in_use measured IDENTICAL to BFC.
		#
		# The environment alone cannot answer this: the allocator is fixed
		# at backend init, so a variable set later changes the string and
		# not the client (job 7882443).  Corroborate against the device.
		_xm = resolve_xla_gpu_memory_env()
		_backend = jax.default_backend()
		try:
			_stats = jax.local_devices()[0].memory_stats()
		except Exception as _exc:
			_stats = None
			print_fn(f"    [mem] memory_stats() unavailable "
			         f"({type(_exc).__name__}: {_exc}); the peak below is "
			         f"whatever the ζ-fit tracker could sample.")
		_pool = classify_xla_pool(_stats, backend=_backend, env=_xm)
		_caveat = _xm.caveat()
		if _pool.disagreement:
			print_fn(f"    *** LORRAX SANITY: {_pool.disagreement} ***")
		# Label the line by the LIVE backend.  On a CPU-backend run that
		# lands on a GPU node (JAX_PLATFORMS=cpu with SLURM still exporting
		# CUDA_VISIBLE_DEVICES) the nvidia-smi fallback reads a GPU this
		# run never used, so calling it a "GPU high-water mark" would be a
		# statement about another job.
		# ``budget_gb`` is 0 when no device memory could be detected (the CPU
		# backend reaches this line too).  Print "n/a" rather than divide.
		_pct = (f"{100 * peak_gb / budget_gb:.0f}%" if budget_gb > 0
		        else "budget unknown")
		if str(_backend).strip().lower() in ("gpu", "cuda", "rocm"):
			_src = ("XLA arena" if _pool.peak_source == "arena"
			        else "nvidia-smi whole-GPU sample")
			print_fn(f"    GPU high-water mark: {peak_gb:.2f} GB / "
			         f"{budget_gb:.2f} GB budget ({_pct})  "
			         f"[source: {_src}]{_caveat}")
		else:
			print_fn(f"    ζ-fit high-water mark: {peak_gb:.2f} GB / "
			         f"{budget_gb:.2f} GB budget ({_pct})  [backend="
			         f"{_backend}: this figure comes from the device "
			         f"memory-stats/nvidia-smi tracker and is NOT this "
			         f"run's host RSS — use LORRAX_MEM_DEBUG=1 for that]")

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
	# CrI3 30 Ry test-bed KNOWN_SANDBOX_ERRORS entry.  The
	# missing-centroids_file_current refusal itself now lives in the
	# bispinor PRE-FLIGHT above (before the charge fit), so this branch
	# gate is only reachable with the path present.
	if cfg.bispinor and getattr(cfg.paths, 'centroids_file_current', None):
		import dataclasses
		from common.wfn_transforms import load_centroids_band_chunked
		from file_io.centroids import load_centroids

		cents_curr_path = cfg.paths.centroids_file_current
		print_fn(f"\n  [bispinor] fitting ζ^{{μ_L=1,2,3}} on current-density "
		         f"centroids: {cents_curr_path}")
		_, cents_curr_idx, n_rmu_curr = load_centroids(
			cents_curr_path, meta.fft_grid)
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
					distributed_cholesky=cfg.backend.distributed_cholesky,
					distributed_lu=cfg.backend.distributed_lu,
					zeta_ridge=cfg.backend.zeta_ridge,
					distributed_zeta_solve=cfg.backend.distributed_zeta_solve,
					gflat_chunk_size=int(chunks.get('gflat_chunk_size', 0)),
					vertex_mu_L=mu_L,
					# Transverse ζ IBZ-write activates whenever the
					# bispinor V_q orchestrator iterates IBZ q's — same
					# gate the charge ζ uses (LORRAX_FORCE_FULL_BZ off).
					# Orbit-closure of the transverse centroid set is
					# checked downstream in ``fit_zeta_to_h5``; failure
					# is loud per the bispinor IBZ requirement.
					write_ibz_only=(bool(cfg.bispinor)
					                and not env_bool(
						                'LORRAX_FORCE_FULL_BZ', False,
						                print_fn=print_fn)),
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
	barrier("zeta_flush")

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
		_use_ibz_bispinor = not env_bool(
			'LORRAX_FORCE_FULL_BZ', False, print_fn=print_fn)
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
			from file_io.zeta_loader import ZetaLoader
			with ZetaLoader(zeta_h5_path, mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zc, \
			     ZetaLoader(zeta_T_paths[0], mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zt1, \
			     ZetaLoader(zeta_T_paths[1], mesh=mesh_xy,
			                backend=cfg.backend.slab_io) as zt2, \
			     ZetaLoader(zeta_T_paths[2], mesh=mesh_xy,
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
		# layout raises.  ``ZetaLoader`` is the V_q reader of record —
		# it serves the writer's per-q WFN.h5-style G-sphere directly.
		from file_io.zeta_loader import ZetaLoader
		with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
			with ZetaLoader(zeta_h5_path, mesh=mesh_xy,
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
			# Clip the μ axis to the LOGICAL extent: files on disk store
			# logical extents so they re-read identically on any process
			# count (G0_all is at the in-memory padded extent, pad
			# entries exact zeros).
			_g0_np = np.asarray(G0_gathered)[..., :int(meta.n_rmu)]
			f.create_dataset('g0_mu', data=_g0_np)
	barrier("g0_write")

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
	_vq0_trace = float(jnp.trace(V_q_raw[0]).real)
	print_fn(f"    V_q=0 trace: {_vq0_trace:.4f}")

	# ── V_q stage gate ────────────────────────────────────────────────
	# Three one-sweep invariants on the tensor every later stage (χ₀, W,
	# Σ_x, Σ_c, the BSE kernel) is built from.  Historically this seam
	# produced a 27 % shift in ``tr V_{q=0}`` between two runs whose V_q
	# is band-window-independent and therefore *must* have been identical
	# — a discrepancy that was only noticed days later, by hand, from log
	# archaeology.  V is a positive-definite Gram matrix in the ISDF
	# basis, so its q=0 trace is positive by construction and its tiles
	# are Hermitian by construction; both are cheap to state.
	from common import sanity
	sanity.check_finite("V_q", V_qmunu, print_fn=print_fn)
	sanity.check_positive("V_q[q=0] trace", _vq0_trace, print_fn=print_fn)
	sanity.check_hermitian("V_q[q=0]", V_q_raw[0], print_fn=print_fn)
	sanity.check_finite("V_q G0 (ζ_μ(G=0) at q=0)", G0, print_fn=print_fn)
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
			from gw.gw_config import SlabIOBackend
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
				# Stage F (restart-tensor write) is UNSHARDED on the
				# allgather backend — ``_slab_io_allgather._to_host``
				# process_allgathers the whole (n_q, μ, μ) tensor onto every
				# rank.  The parallel-HDF5 backends write per-rank hyperslabs
				# and cost the sharded amount instead.
				slab_io_replicates=(
					cfg.backend.slab_io == SlabIOBackend.H5PY_ALLGATHER),
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
			#
			# The parse used to be a bare presence test, so *every* non-empty
			# value exited — including ``LORRAX_EXIT_AFTER_ZETA=0``.  A debug
			# knob set to "off" therefore ended a production run with
			# ``SystemExit(0)``: the worst possible failure shape, because
			# rc=0 with a truncated output is indistinguishable from
			# completion to anything downstream.  ``env_bool`` gives ``0``,
			# ``off``, ``false``, ``no`` (any case) their obvious meaning and
			# announces anything it does not recognise.
			if env_bool("LORRAX_EXIT_AFTER_ZETA", False, print_fn=print0):
				if jax.process_index() == 0:
					# Loud and machine-greppable: this exit is rc=0 by
					# design (runtime's fail-fast excepthook documents
					# SystemExit(0) as intentional), so the LOG is the only
					# place a consumer can tell an early exit from a
					# finished run.
					print("*** LORRAX EARLY EXIT: LORRAX_EXIT_AFTER_ZETA=1 "
					      "— stopping after fit_zeta.  V_q, W, and Σ were "
					      "NOT computed; this run produced no self-energy. "
					      "***", flush=True)
					import sys as _sys
					print("*** LORRAX EARLY EXIT (LORRAX_EXIT_AFTER_ZETA) ***",
					      file=_sys.stderr, flush=True)
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
			if env_bool("LORRAX_MEM_DEBUG", False, print_fn=print0):
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
				n_rmu_logical=int(meta.n_rmu),
				V_qmunu=V_qmunu, G0_mu_nu=G0, enk_full=enk_full,
				init_W0=True, mesh=mesh_xy, backend=cfg.backend.slab_io,
				mode="w", kgrid=tuple(int(v) for v in meta.kgrid),
				# Stamp the band window + n_rmu so a later restart
				# under a CHANGED window fails loudly instead of
				# silently misindexing Sigma (job 7874375).
				band_slices=band_slices,
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

			# Append ψ to the now-open restart file.  Bispinor: also the
			# σ^B-side transverse-centroid ψ (per-channel second dataset),
			# clipped to ITS logical extent — this is what makes a
			# bispinor restart round-trip (the loader re-pads and
			# ``prepare_isdf_and_wavefunctions`` rebuilds the second
			# bundle from it; without it Σ^B would be silently dropped).
			write_restart_state_to_h5(
				tensors_filename,
				n_rmu_logical=int(meta.n_rmu),
				psi_full_y=wfns.psi_yr, mesh=mesh_xy,
				backend=cfg.backend.slab_io, mode="a",
				psi_full_y_transverse=(
					wfns_transverse.psi_yr
					if wfns_transverse is not None else None),
				n_rmu_transverse_logical=(
					int(transverse_wfn_data['meta'].n_rmu)
					if transverse_wfn_data is not None else None),
			)
			# Stamp the centroid tables' CONTENT hashes so a restart can
			# verify the quadrature points, not just their counts (see
			# ``_centroid_table_md5``; audit fix/zq 2026-07-28).  Rank 0
			# only, after the collective writer released the file; a
			# failed stamp is non-fatal — the restart-side guard then
			# warns about the missing attr instead of verifying.
			if jax.process_index() == 0:
				try:
					with h5py.File(tensors_filename, 'a') as _f:
						_f.attrs['centroids_charge_md5'] = (
							_centroid_table_md5(centroid_indices))
						if transverse_wfn_data is not None:
							_f.attrs['centroids_transverse_md5'] = (
								_centroid_table_md5(
									transverse_wfn_data['centroid_indices']))
				except Exception as exc:
					print0(f"    [restart stamp] centroid content hashes "
					       f"not stamped ({exc}); a restart will warn "
					       f"instead of verifying the centroid tables.")
			barrier("restart_centroid_stamp")
		save_restart_state_per_proc(
			os.path.join(tmp_dir, "isdf_tensors"),
			V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices,
				n_rmu_logical=int(meta.n_rmu))
			V_qmunu = rs.V_qmunu
			print0("  Loaded restart tensors from H5.")
			# Restart is the seam where "rc=0 but garbage" was born (job
			# 7874375: a changed band window silently reused tensors built
			# under the old one).  The band-window attrs guard inside
			# ``load_restart_state_from_h5`` covers provenance; these
			# gates cover the *content* — a truncated/partially-written
			# restart file from a crashed run reads back as zeros or NaN
			# and would otherwise flow straight into Σ.
			from common import sanity
			sanity.check_finite("restart V_q", V_qmunu, print_fn=print0)
			sanity.check_positive(
				"restart V_q[q=0] trace",
				float(jnp.trace(V_qmunu[0]).real), print_fn=print0)
			sanity.check_finite("restart ψ (psi_full_y)", rs.psi_rmu_Y,
			                    print_fn=print0)
			sanity.check_finite("restart E_nk", rs.enk_full, print_fn=print0)
			# Centroid-table CONTENT guard (pattern #10; audit fix/zq
			# 2026-07-28).  The band-window/n_rmu attrs pin the SHAPES of
			# the restart tensors; these md5 stamps (written by the
			# non-restart path above) pin the quadrature POINTS.  Old
			# restart files predating the stamp get a LOUD warning naming
			# the gap, not a refusal.
			_stamped = {}
			try:
				with h5py.File(tensors_filename, 'r') as _f:
					for _a in ('centroids_charge_md5',
					           'centroids_transverse_md5'):
						if _a in _f.attrs:
							_stamped[_a] = str(_f.attrs[_a])
			except Exception as exc:
				print0(f"  [restart guard] could not read centroid hash "
				       f"attrs from {tensors_filename} "
				       f"({type(exc).__name__}: {exc}).")
			_have_c = _stamped.get('centroids_charge_md5')
			if _have_c is None:
				print0(
					f"  *** LORRAX SANITY: {tensors_filename} carries no "
					f"'centroids_charge_md5' attr — it predates the "
					f"centroid-content stamp (2026-07-28), so the charge "
					f"centroid TABLE cannot be verified against this "
					f"run's centroids_file; only counts and band windows "
					f"are checked.  If the centroid file may have been "
					f"regenerated since these tensors were written, rerun "
					f"with restart = false. ***")
			elif _have_c != _centroid_table_md5(centroid_indices):
				raise ValueError(
					f"restart: {tensors_filename} was written for a "
					f"DIFFERENT charge centroid table (md5 {_have_c} on "
					f"disk vs {_centroid_table_md5(centroid_indices)} "
					f"from this run's centroids_file).  Same count, "
					f"different points ⇒ ψ/V_q sampled at the wrong r_μ "
					f"(silently wrong physics).  Set restart = false, or "
					f"restore the original centroid file.")
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_rmu_Y=rs.psi_rmu_Y, psi_rmuT_X=rs.psi_rmuT_X,
				enk_full=rs.enk_full, print_fn=print0)
			# Bispinor restart: the transverse-centroid ψ round-trips
			# through the per-channel ``psi_full_y_transverse`` dataset
			# (written 2026-07-27+).  Anything missing or mismatched is
			# a LOUD refusal — before this, ``wfns_transverse=None``
			# flowed into the Σ kernels whose Σ^B fold-in is a silent
			# no-op on None (rc=0, wrong physics: Σ^B dropped).
			wfns_transverse = None
			if cfg.bispinor:
				if getattr(rs, 'psi_rmu_Y_transverse', None) is None:
					raise ValueError(
						f"bispinor restart: {tensors_filename} has no "
						f"'psi_full_y_transverse' dataset.  It was written "
						f"either by a scalar run or by a LORRAX predating "
						f"the bispinor restart round-trip (2026-07-27).  "
						f"Σ^B needs ψ at the transverse centroids; rerun "
						f"with restart = false to rebuild the tensors "
						f"(the ζ fits and v_q_bispinor.h5 will be "
						f"regenerated).")
				# Provenance: the on-disk transverse centroid COUNT and
				# CONTENT (md5 stamp) must match the run's
				# centroids_file_current (pattern #10 — the artifact
				# outlives the config that made it).
				if not getattr(cfg.paths, 'centroids_file_current', None):
					raise ValueError(
						"bispinor restart requires centroids_file_current "
						"in the input file (same requirement as the "
						"non-restart bispinor path).")
				from file_io.centroids import load_centroids as _load_cent
				_, _cent_T_idx_now, _n_rmu_curr_now = _load_cent(
					cfg.paths.centroids_file_current, meta.fft_grid)
				if int(_n_rmu_curr_now) != int(rs.n_rmu_transverse_disk):
					raise ValueError(
						f"bispinor restart: {tensors_filename} stores the "
						f"transverse ψ for n_rmu_T="
						f"{int(rs.n_rmu_transverse_disk)} centroids, but "
						f"centroids_file_current "
						f"({cfg.paths.centroids_file_current}) now has "
						f"{int(_n_rmu_curr_now)}.  The σ^B quadrature "
						f"basis differs; set restart = false (or restore "
						f"the original transverse centroid file).")
				# Content check — same count does NOT mean same points
				# (audit fix/zq 2026-07-28; ``_stamped`` read above).
				_have_t = _stamped.get('centroids_transverse_md5')
				if _have_t is None:
					print0(
						f"  *** LORRAX SANITY: {tensors_filename} carries "
						f"no 'centroids_transverse_md5' attr — it "
						f"predates the centroid-content stamp "
						f"(2026-07-28), so the transverse centroid TABLE "
						f"cannot be verified; only its count "
						f"({int(rs.n_rmu_transverse_disk)}) was checked.  "
						f"A regenerated centroids_file_current with the "
						f"same count would be consumed SILENTLY (Σ^B at "
						f"the wrong r_μ).  If in doubt, rerun with "
						f"restart = false. ***")
				elif _have_t != _centroid_table_md5(_cent_T_idx_now):
					raise ValueError(
						f"bispinor restart: {tensors_filename} was "
						f"written for a DIFFERENT transverse centroid "
						f"table (md5 {_have_t} on disk vs "
						f"{_centroid_table_md5(_cent_T_idx_now)} from "
						f"centroids_file_current).  Same count "
						f"({int(_n_rmu_curr_now)}), different points ⇒ "
						f"Σ^B evaluated with ψ sampled at the wrong r_μ "
						f"(silently wrong physics).  Set restart = false, "
						f"or restore the original transverse centroid "
						f"file.")
				sanity.check_finite(
					"restart transverse ψ (psi_full_y_transverse)",
					rs.psi_rmu_Y_transverse, print_fn=print0)
				wfns_transverse = build_wavefunction_bundle(
					wfn, sym, meta, band_slices, mesh_xy,
					psi_rmu_Y=rs.psi_rmu_Y_transverse,
					psi_rmuT_X=rs.psi_rmuT_X_transverse,
					enk_full=rs.enk_full, print_fn=print0)
				print0(f"  [bispinor] σ^B-side Wfns rebuilt from restart "
				       f"(n_rmu_T={int(rs.n_rmu_transverse_disk)} "
				       f"transverse centroids)")

	return SimpleNamespace(
		V_qmunu=V_qmunu,
		wf_bundle=wfns,
		wf_bundle_transverse=wfns_transverse,
	)

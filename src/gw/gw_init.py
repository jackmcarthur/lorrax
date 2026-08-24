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
from common import rank_criterion
from common import spectral_closure
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
	refuse_unsupported_bispinor_tt_head_correction,
	refuse_unsupported_low_mem_bands,
	resolve_xla_gpu_memory_env,
)

# ── The ζ file's DOOR ────────────────────────────────────────────────────
# ``zeta_q.h5`` (and its bispinor siblings) has exactly one owner, the
# ``zeta_loader`` service package under ``services/``.  This module reaches
# it through the TOP-LEVEL package only: ``zeta_loader.format`` /
# ``zeta_loader.loader`` are past-the-door edges and
# ``tests/test_layering.py`` fails on them.
#
# ``ffi._services.ensure_on_path()`` is why the import below resolves in a
# bare launch — nothing in the launch chain knows ``services/`` exists
# (``lx`` rewrites the container PYTHONPATH to exactly ``<checkout>/src``).
# Transitional plumbing with an owner decision behind it; see
# ``src/ffi/_services.py``.
#
# THREE NAMES, THREE FORMER RAW-h5py SITES.  ``probe_zeta_file`` replaced
# the two hand-written copies of the dataset-name/μ-axis dispatch below
# (``_check_zeta_h5_matches_basis`` and ``_zeta_reuse_ok``) — that tuple is
# spelled ONCE now, in the service, and this comment deliberately does not
# spell it a fourth time; ``ZetaLoader`` replaced the raw ``n_rmu_T`` shape
# read; ``write_g0_mu`` replaced the raw ``del``+``create_dataset`` append.
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from zeta_loader import (                                        # noqa: E402
	ZetaLoader, probe_zeta_file, write_g0_mu)
# The vcoul door, for the ONE thing this module needs from it: the
# Cartesian reciprocal rows, taken as a geometry rather than written out as
# ``blat * bvec`` at the V_q call site below.
from vcoul import CoulombGeometry                                # noqa: E402


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
	   written to protect.  Both names are probed now, and the probe is
	   :func:`zeta_loader.probe_zeta_file` rather than a copy of the dispatch
	   written out here: that copy, and its twin in :func:`_zeta_reuse_ok`,
	   are how one truth came to be spelled three times and two of the three
	   were wrong for months.
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
	# ONE open, one pass, and it NEVER RAISES — which is the whole reason
	# this guard can run before the fit that is about to overwrite the
	# file.  ``probe.error`` is the ``f"{type(exc).__name__}: {exc}"`` this
	# site used to format itself, so the print below is unchanged.
	probe = probe_zeta_file(zeta_h5_path)
	if not probe.readable:
		# Unreadable/partial file: say so, then let the writer deal with it.
		# ``readable=False`` with the file present means exactly what the
		# ``except Exception`` here used to catch.
		print_fn(f"  [zeta guard] could not read {zeta_h5_path} "
		         f"({probe.error}); continuing.")
		return
	existing = probe.mu_extent          # μ off the ζ BLOCK (dispatch: door)
	zeta_done = probe.zeta_done         # isdf_header/zeta_is_done
	header_grid = probe.r_mu_fft_idx    # isdf_header/centroids/r_mu_fft_idx
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
                         band_norms, vertex_mu_L=0,
                         transverse_identity=None):
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

	``wfn_file`` is the RESOLVED path (``realpath``, not ``abspath``) and
	``wfn_bytes`` its size, which together identify the source WFN.h5
	without reading it.  Resolving matters because the fleet stages each
	leg behind its own directory or symlink: under ``abspath`` the same
	file arrived spelled differently on every launch, and the reuse check
	read that as a changed input.  ``_same_wfn_file`` re-resolves both
	sides at comparison time as well, so a stamp written before this line
	— which holds an unresolved path — is still read correctly.  mtime is
	deliberately NOT included: copying/restoring a WFN.h5 changes mtime
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

	``meta`` / ``vertex_mu_L`` select WHICH ζ file this stamp describes.
	A bispinor run writes FOUR ζ files from one configuration — the
	charge ζ (``meta``, ``vertex_mu_L=0``) and three transverse ζ
	(``meta_T``, ``vertex_mu_L ∈ {1,2,3}``) — so the same builder is
	called once per file with that file's own μ extent and vertex.  The
	stamps then differ in exactly the two entries that make the files
	different, and a ζ_T stamped for one vertex can never be reused as
	another.

	``transverse_identity`` pins the transverse CHANNEL on every stamp,
	including the charge one: the transverse centroid count, that
	table's content hash, and the two knobs that set the transverse
	solve gauge (``distributed_lu`` and the resolved transverse solver
	kind).  It is REQUIRED for a bispinor run — a bispinor ζ set whose
	stamp did not name its transverse basis could be reused by a rerun
	that changed ``centroids_file_current``, which is the σ^B analogue
	of pattern #10.  It is ignored (all four keys collapse to ``None``)
	for a non-bispinor run, where no transverse channel exists and the
	keys must not force a spurious refit of any existing charge-only ζ.
	"""
	import json
	if bool(cfg.bispinor) and not transverse_identity:
		raise ValueError(
			"_zeta_fit_provenance: cfg.bispinor is set but no "
			"transverse_identity was supplied.  A bispinor ζ stamp that "
			"omits the transverse centroid table / solver identity would "
			"let a rerun with a DIFFERENT centroids_file_current reuse "
			"this fit (Σ^B evaluated at the wrong r_μ).  Pass the dict "
			"built in fit_zeta's bispinor pre-flight.")
	_ti = dict(transverse_identity or {}) if bool(cfg.bispinor) else {}
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
		# GAUGE tier of the charge-channel factor (zeta audit 2026-08-01):
		# the `distributed` tier's block-cyclic pzheevd is a different,
		# equally valid gauge (~kappa*eps vs the whole-tile eigh), so a
		# zeta fit under one tier must not be silently reused by a rerun
		# under the other.  `replicated` and `per_q` are back-solve GATHER
		# granularities over the SAME whole-tile factor (bit-identical),
		# and `auto` never resolves to `distributed` — so exactly two
		# gauge classes exist and both collapse here.  The schema is NOT
		# bumped: `_zeta_reuse_ok` treats a stamp MISSING this key as
		# legacy replicated-gauge (every pre-2026-08 zeta was), keeping
		# old run dirs usable.
		'distributed_zeta_solve': (
			'distributed'
			if str(cfg.backend.distributed_zeta_solve).strip().lower()
			== 'distributed' else 'replicated'),
		# TRANSVERSE solve family + cut (2026-08-01, same reader-matrix
		# idiom as the tier key above): recorded so a family change is
		# never silently absorbed by reuse.  Collapsed to the inert
		# canonical ('ridge', None) when the run has no transverse
		# channel (non-bispinor) — the keys do not change the CHARGE ζ
		# numerically, so a non-bispinor deck toggling them must not
		# force a spurious refit.  τ is likewise collapsed to None under
		# the ridge family (it is not read there).  A stamp MISSING these
		# keys is a legacy ridge-family fit (every pre-2026-08 ζ was);
		# `_zeta_reuse_ok` allows reuse for a ridge-family rerun and
		# refits on a real family mismatch.
		'transverse_zeta_solve': (
			str(cfg.backend.transverse_zeta_solve).strip().lower()
			if cfg.bispinor else 'ridge'),
		'transverse_zeta_rcond': (
			float(cfg.backend.transverse_zeta_rcond)
			if (cfg.bispinor
			    and str(cfg.backend.transverse_zeta_solve).strip().lower()
			    == 'rank_truncate') else None),
		# TRANSVERSE CHANNEL identity (2026-08-04, same reader-matrix
		# idiom as the two key generations above).  ζ reuse used to be
		# switched off for the whole bispinor run, so none of this had
		# anywhere to be recorded; with reuse live, everything that
		# changes a ζ_T has to be here or a wrong ζ_T gets reused.
		#   n_rmu_transverse         — the transverse μ extent.
		#   centroids_transverse_md5 — that table's CONTENT (same hash
		#       `_centroid_table_md5` stamps on the restart tensors): a
		#       regenerated centroids_file_current with the SAME count
		#       but different points is pattern #10 and passes a
		#       count-only check.
		#   distributed_lu           — the transverse LU backend.  A ζ_T
		#       fit under `scalapack` (block-cyclic gauge) must not be
		#       reused by an `off` rerun (per-q jnp.linalg.solve) and
		#       vice versa.  This is the RESOLVED deck value: gw_config
		#       already demotes `auto`→`off` on a CPU backend, so the
		#       recorded string is what the fit ran.
		#   transverse_solver_kind   — what `_resolve_solver_kind_transverse`
		#       actually returned ('lu' | 'scalapack_lu' | 'cusolvermp_lu'
		#       | 'transverse_rank_truncate').  Recorded IN ADDITION to
		#       the two knobs above because on a GPU mesh `auto` resolves
		#       by mesh shape, and those two resolutions are genuinely
		#       different gauges.  This is the one place a device-count
		#       dependence is deliberately admitted into the stamp: the
		#       exclusion at the top of this docstring covers quantities
		#       ζ is INVARIANT under, and this is not one of them.
		# All four collapse to None on a non-bispinor deck (no transverse
		# channel ⇒ inert), so a charge-only rerun over a pre-2026-08-04
		# stamp reuses under the legacy-missing-key rule below.
		'n_rmu_transverse':     (int(_ti['n_rmu']) if _ti else None),
		'centroids_transverse_md5': (
			str(_ti['centroids_md5']) if _ti else None),
		'distributed_lu':       (str(_ti['distributed_lu']) if _ti else None),
		'transverse_solver_kind': (
			str(_ti['solver_kind']) if _ti else None),
		'gamma_contract_mode':  str(cfg.backend.gamma_contract_mode),
		'write_ibz_only':       bool(write_ibz_only),
		'vertex_mu_L':          int(vertex_mu_L),
		'band_norms':           bn,
		'fft_grid':             [int(x) for x in np.asarray(meta.fft_grid).reshape(3)],
		'ecutwfc':              round(float(wfn.ecutwfc), 9),
		'ecutrho':              round(float(wfn.ecutrho), 9),
		# RESOLVED, not as-typed.  ``abspath`` only prepends the cwd; it
		# leaves every symlink in place, so the same WFN.h5 reached through
		# a per-run staging link stamped a different string on every launch
		# and the comparison below refit for a rename (2026-08-09).  See
		# :func:`_same_wfn_file` for the identity this key now carries and
		# for how a stamp written before this line is still read correctly.
		'wfn_file':             os.path.realpath(wfn_path) if wfn_path else '',
		'wfn_bytes':            wfn_bytes,
	}
	return json.dumps(prov, sort_keys=True)


def _wfn_path_identity(path):
	"""``(realpath, size, mtime_ns)`` for one recorded ``wfn_file`` string.

	Never raises.  A path taken off an old stamp routinely no longer
	exists — the run directory was cleaned, the staging link was torn
	down — and that is "cannot prove anything about this spelling", not
	an error.  The stat fields are ``None`` in that case and the caller
	decides what silence means.
	"""
	real = os.path.realpath(path) if path else ''
	try:
		st = os.stat(real)
	except OSError:
		return real, None, None
	return real, int(st.st_size), int(st.st_mtime_ns)


def _same_wfn_file(old_path, new_path, *, old_bytes=None, new_bytes=None):
	"""Do two recorded ``wfn_file`` spellings name the SAME WFN.h5?

	Returns ``(verdict, why)``; ``why`` is a sentence naming BOTH paths,
	because the only useful thing to say about a refusal here is which
	two files the run thinks it is choosing between.

	A path string is not an identity.  The fleet stages each leg's inputs
	behind its own directory or symlink, so one WFN.h5 arrives spelled a
	different way on every launch; comparing the strings charged a full
	ζ re-fit (16.81 GiB on the production deck) for a rename, on runs
	where every spelling resolved to the same bytes.  Identity is
	therefore taken in two steps, cheapest first:

	1. **Resolved name.**  ``os.path.realpath`` collapses symlinks and
	   ``..`` segments, which is every case the staging layout produces.
	2. **Same inode.**  Two spellings can survive step 1 and still be one
	   file — a bind mount, a hard link, two mount points onto the same
	   backing store.  ``os.path.samefile`` (device + inode) settles
	   those, and costs one ``stat`` each.

	Then the CONTENT stamp, which answers the different question: the
	name may be stable while the bytes behind it are not.  ``wfn_bytes``
	(the size recorded at fit time, passed in here) is compared against
	the size this run recorded for its own input, and the resolved file's
	size and mtime go into the refusal message so a real replacement is
	legible rather than mysterious.  The bound is the same one
	:func:`_zeta_fit_provenance` already declares: a same-size,
	same-path replacement is not caught, and a spurious multi-hour refit
	was judged the worse outcome than that contrived case.
	"""
	if not old_path or not new_path:
		return False, ("one of the two stamps records no WFN path at all "
		               f"(on-disk stamp {old_path!r}, this run {new_path!r})")
	old_real, old_size, old_mtime = _wfn_path_identity(old_path)
	new_real, new_size, new_mtime = _wfn_path_identity(new_path)
	_sizes_known = old_bytes is not None and new_bytes is not None
	if _sizes_known and int(old_bytes) != int(new_bytes):
		return False, (
			f"the on-disk stamp was fit from {old_path!r} at "
			f"{old_bytes} bytes and this run reads {new_path!r} at "
			f"{new_bytes} bytes — different files")
	if old_real != new_real:
		# Different resolved names, but possibly one file underneath.
		try:
			if os.path.samefile(old_real, new_real):
				return True, (f"{old_path!r} and {new_path!r} resolve to "
				              f"different names ({old_real!r}, {new_real!r}) "
				              f"but to the SAME file (device+inode)")
		except OSError:
			pass
		return False, (
			f"they name different files: the on-disk stamp was fit from "
			f"{old_path!r} (resolves to {old_real!r}, "
			f"{'missing' if old_size is None else f'{old_size} bytes'}) and "
			f"this run reads {new_path!r} (resolves to {new_real!r}, "
			f"{'missing' if new_size is None else f'{new_size} bytes'})")
	if new_size is None:
		# Both spell the same resolved name and neither can be stat'd.
		# The names agree and the recorded sizes agree, which is every
		# check this door has; say so rather than refit on absence.
		return True, (f"{old_path!r} and {new_path!r} both resolve to "
		              f"{new_real!r}, which is not present on this node — "
		              f"the recorded sizes agree, so the spellings are read "
		              f"as the same file")
	return True, (f"{old_path!r} and {new_path!r} both resolve to "
	              f"{new_real!r} ({new_size} bytes, mtime_ns {new_mtime})")


def _zeta_reuse_ok(zeta_h5_path, provenance_json, centroid_fft_idx,
                   print_fn=print, *, n_rmu_expected=None):
	"""Can we skip the ζ fit and reuse ``zeta_h5_path`` as-is?

	Returns ``True`` only when EVERY one of these holds:

	  1. ``LORRAX_FORCE_REFIT`` is not set to a truthy value;
	  2. the file exists and its ``isdf_header`` is readable;
	  3. ``zeta_is_done`` is True (the writer flipped it after the last
	     H5Dwrite drained — a crashed fit leaves it False);
	  4. ``fit_provenance`` is present AND byte-identical to this run's —
	     with TWO named exceptions, both announced when they fire.  The
	     legacy-key table below: a stamp that differs ONLY by keys added
	     after it was written is read as having run each of those keys'
	     legacy value, so old run dirs stay reusable by a rerun that asks
	     for exactly those values while any other request refits, naming
	     the key.  And path SPELLING: ``wfn_file`` records a name, and one
	     file has many names, so two spellings that resolve to the same
	     file are not a difference (:func:`_same_wfn_file`) — while two
	     that resolve to different files refit, naming BOTH paths;
	  5. the on-disk centroid table equals this run's centroid indices;
	  6. (``n_rmu_expected`` given) the ζ DATASET's μ extent equals it.
	     The header and the dataset are written by two different calls,
	     so a file can carry a good header over a ζ of the wrong shape.
	     :func:`_check_zeta_h5_matches_basis` RAISES on that for the
	     charge file; here the same probe only declines to reuse,
	     because a transverse ζ from a different centroid set is a
	     legitimate thing to overwrite and must not kill the run.

	Anything unexpected — missing attr, unreadable file, legacy header —
	returns False, i.e. REFIT.  Every failure mode costs compute; none
	costs correctness.  That asymmetry is the whole design: this cache
	sits in front of a multi-hour step whose silent misuse produced a
	−135 eV QP gap once already (job 7874375, the restart-window bug).
	"""
	# Canonical boolean grammar (gw_config.env_bool, imported at module
	# scope with the rest of this layer's env helpers — this site used to
	# lazily import isdf.core._env_bool, the ONE other parser, which P1.3
	# retired): under the old hand-rolled
	# ``not in ('', '0', 'false', 'False')`` parse, ``=off``/``=no``
	# counted as truthy and forced a multi-hour refit (audit fix/zq
	# 2026-07-28).
	if env_bool('LORRAX_FORCE_REFIT', False):
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
			# THREE distinct ways two stamps disagree, kept apart because
			# they get different verdicts (2026-08-04).  The earlier code
			# collapsed them into one value-difference list over the key
			# UNION, which silently mis-handled a key ADDED with a value
			# equal to its legacy default: `old.get(k) is None == new[k]`
			# put the key in NEITHER list, so the difference list came out
			# empty while the JSON strings still differed — and an empty
			# list fell through to the generic "fit under DIFFERENT
			# inputs" refit with an EMPTY "Changed:" detail.  That is a
			# spurious refit of every pre-2026-08-04 charge ζ, i.e. the
			# exact outcome this whole branch exists to prevent.  Caught
			# by tests/test_zeta_provenance_matrix.py.
			_added = sorted(set(new) - set(old))       # stamp predates key
			_dropped = sorted(set(old) - set(new))     # stamp is NEWER
			diff = sorted(k for k in (set(old) & set(new))
			              if old[k] != new[k])
			# PATH SPELLING IS NOT FILE IDENTITY (2026-08-09).  ``wfn_file``
			# is the one key whose value is a NAME rather than a number, and
			# a name has more than one spelling: the fleet stages each leg
			# behind its own directory or symlink, so the same WFN.h5 landed
			# in this list on every comparison and every such run paid a full
			# 16.81 GiB re-fit for a rename.  Resolve the two spellings and
			# ask what they actually name; a genuine difference stays in the
			# list (and is refused below naming BOTH paths), so this only
			# ever removes a difference that was never real.  Done BEFORE the
			# legacy-key exception so that a stamp which both predates a key
			# AND was written under a different spelling still reuses.
			if 'wfn_file' in diff:
				_same, _why = _same_wfn_file(
					old.get('wfn_file'), new.get('wfn_file'),
					old_bytes=old.get('wfn_bytes'),
					new_bytes=new.get('wfn_bytes'))
				if _same:
					print_fn(f"    [zeta reuse] {zeta_h5_path}: the WFN path "
					         f"differs only in SPELLING — {_why}; not a "
					         f"reason to refit.")
					diff.remove('wfn_file')
				else:
					print_fn(f"    [zeta reuse] {zeta_h5_path} was fit from a "
					         f"DIFFERENT WFN — {_why} — refitting.")
					return False
			detail = "; ".join(
				[f"{k}: on-disk={old.get(k)!r} now={new.get(k)!r}"
				 for k in diff]
				+ ([f"keys ADDED since the stamp: {', '.join(_added)}"]
				   if _added else [])
				+ ([f"keys the stamp has and this LORRAX does not write: "
				    f"{', '.join(_dropped)}"] if _dropped else []))
		except Exception:
			old = new = None
			diff = _added = _dropped = None
			detail = "(provenance unparseable)"
		# Legacy-stamp compatibility (owner-approved, 2026-08-01): a stamp
		# MISSING a key predates that key's feature, and every pre-feature
		# fit ran the feature's legacy value.  When the only differences
		# are such missing keys, reuse is allowed iff this run requests
		# exactly the legacy value for each (one-line notice); requesting
		# anything else IS a real mismatch — refit, naming the key.  The
		# schema number stays at 1 on purpose: bumping it would force a
		# refit of every existing on-disk zeta, which this branch exists
		# to avoid.  Keys and their implied legacy values:
		#   distributed_zeta_solve  'replicated'  (the distributed tier's
		#       block-cyclic eigh is a different gauge — 2026-08-01)
		#   transverse_zeta_solve   'ridge'       (the rank_truncate
		#       family is a different transverse solve — 2026-08-01)
		#   transverse_zeta_rcond   None          (unused under ridge)
		#   n_rmu_transverse         None  ┐ 2026-08-04: ζ reuse was
		#   centroids_transverse_md5 None  │ charge-channel-only before,
		#   distributed_lu           None  │ so a stamp lacking these
		#   transverse_solver_kind   None  ┘ four was written by a run
		#       whose reuse decision never consulted a transverse
		#       channel.  A non-bispinor rerun requests None for all
		#       four and reuses (they are inert without a transverse
		#       channel); a BISPINOR rerun requests real values and
		#       refits — the only safe reading, because such a stamp
		#       carries no evidence about the ζ_T files beside it and
		#       those ζ_T carry no stamp of their own at all.
		_LEGACY_KEY_DEFAULTS = {
			'distributed_zeta_solve': 'replicated',
			'transverse_zeta_solve': 'ridge',
			'transverse_zeta_rcond': None,
			'n_rmu_transverse': None,
			'centroids_transverse_md5': None,
			'distributed_lu': None,
			'transverse_solver_kind': None,
		}
		# The exception applies ONLY when the difference is entirely
		# "this stamp predates some keys": every key the two stamps share
		# agrees, nothing the stamp carries has been dropped, and every
		# added key has a declared legacy meaning.  A stamp carrying a key
		# this LORRAX no longer writes came from a NEWER build and gets no
		# exception — its meaning is unknown here.
		# The two stamps' BYTES differ but, after the path-spelling
		# resolution above, nothing they say differs.  This needs its own
		# arm: the legacy exception below requires a non-empty ``_added``,
		# so without this an all-spelling difference would fall through to
		# the generic refit with an EMPTY "Changed:" detail — the same
		# shape as the 2026-08-04 regression this branch already carries a
		# test for, and the same outcome (a refit for nothing).
		_only_spelling = (old is not None and not diff and not _added
		                  and not _dropped)
		_legacy_shaped = (
			old is not None and not diff and not _dropped and _added
			and all(k in _LEGACY_KEY_DEFAULTS for k in _added))
		if _only_spelling:
			pass
		elif _legacy_shaped:
			_mismatch = [k for k in _added
			             if new.get(k) != _LEGACY_KEY_DEFAULTS[k]]
			if not _mismatch:
				print_fn(f"    [zeta reuse] {zeta_h5_path}: legacy stamp "
				         f"(missing {', '.join(_added)}) — this run requests "
				         f"the legacy value(s) those imply; reuse allowed.")
			else:
				_d2 = "; ".join(
					f"{k}: legacy-implied={_LEGACY_KEY_DEFAULTS[k]!r} "
					f"requested={new.get(k)!r}" for k in _mismatch)
				print_fn(
					f"    [zeta reuse] {zeta_h5_path}: legacy stamp lacks "
					f"{', '.join(_added)} and this run requests a DIFFERENT "
					f"value than the legacy fit implies ({_d2}) — "
					f"refitting.")
				return False
		else:
			print_fn(f"    [zeta reuse] {zeta_h5_path} was fit under "
			         f"DIFFERENT inputs — refitting.  Changed: {detail}")
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
	if n_rmu_expected is not None:
		# Header ≠ dataset: the two are written by separate calls
		# (write_isdf_header, then the SlabIO create/write), so a file
		# can pass every check above and still hold a ζ of the wrong
		# shape — e.g. a run killed after the header write, or a
		# transverse ζ left over from a different centroids_file_current.
		# Same door, same never-raising contract as
		# :func:`_check_zeta_h5_matches_basis` — this site and that one
		# were the two hand-written copies of the layout dispatch, and the
		# probe is the one copy now.  ``probe.error`` is the
		# ``f"{type(exc).__name__}: {exc}"`` this site used to format.
		_probe = probe_zeta_file(zeta_h5_path)
		if not _probe.readable:
			print_fn(f"    [zeta reuse] {zeta_h5_path}: ζ dataset unreadable "
			         f"({_probe.error}) — refitting.")
			return False
		_ext = _probe.mu_extent
		if _ext is None:
			print_fn(f"    [zeta reuse] {zeta_h5_path} has an isdf_header but "
			         f"NO ζ dataset (neither zeta_q_G nor zeta_q) — "
			         f"refitting.")
			return False
		if _ext != int(n_rmu_expected):
			print_fn(f"    [zeta reuse] {zeta_h5_path} holds a ζ at "
			         f"n_mu={_ext} but this channel has "
			         f"n_mu={int(n_rmu_expected)} — refitting.")
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


def _transverse_wfn_data(wfn, sym, meta_T, cent_T_idx, cfg, mesh_xy,
                         band_slices, chunks):
	"""Sample ψ at the TRANSVERSE centroid set and package it for σ^B.

	This is the whole of ``transverse_wfn_data``: ψ at r_{μ_T} over the
	full band window, plus the transverse ``meta`` and centroid table.
	It is a pure function of (WFN.h5, symmetry, transverse centroids,
	band window) — the ζ_T fit contributes NOTHING to it, which is why
	the ζ-reuse path can rebuild it instead of refitting.

	ONE call site per path (fit and reuse) on purpose: the reuse leg
	must produce bit-identical ψ to the fit leg, and the cheapest way
	to guarantee that is for both to run the same code.

	``low_mem_bands = true`` (2026-08-23): ALSO converts to the two-face
	carrier here, SAME ``PSI_MUN_SPEC``/``PSI_NMU_SPEC`` build path the
	charge channel uses, and drops the single-axis copies -- once, in
	this one function, so BOTH callers (the fresh-fit bispinor loop and
	the ζ-reuse early return) get the face carrier identically instead
	of each converting it their own way.  ``psi_rmu_Y``/``psi_rmuT_X``
	are ``None`` in the returned dict in this case;
	``psi_mun_fresh``/``psi_nmu_fresh`` carry the face arrays instead.
	"""
	from common.wfn_transforms import load_centroids_band_chunked
	with timing.section("gw_jax.load_centroid_wfns_current"):
		psi_curr_rmu_Y, psi_curr_rmuT_X = load_centroids_band_chunked(
			wfn, sym, meta_T, cent_T_idx, cfg.bispinor, mesh_xy,
			band_range=band_slices.full_range,
			band_chunk_size=chunks['band_chunk'],
		)
	psi_mun_fresh_T = None
	psi_nmu_fresh_T = None
	if cfg.memory.low_mem_bands:
		from jax.sharding import NamedSharding
		from .wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC
		with mesh_xy:
			psi_nmu_fresh_T = jax.lax.with_sharding_constraint(
				psi_curr_rmu_Y, NamedSharding(mesh_xy, PSI_NMU_SPEC))
			psi_mun_fresh_T = jax.lax.with_sharding_constraint(
				jnp.conj(psi_curr_rmuT_X).transpose(0, 3, 1, 2),
				NamedSharding(mesh_xy, PSI_MUN_SPEC))
		del psi_curr_rmu_Y, psi_curr_rmuT_X
		psi_curr_rmu_Y = None
		psi_curr_rmuT_X = None
	# Keeping these arrays alive across the return is intentional —
	# they are the only way the σ^B kernel can sample ψ at r_{μ_T}.
	return {
		'psi_rmu_Y':        psi_curr_rmu_Y,
		'psi_rmuT_X':       psi_curr_rmuT_X,
		'meta':             meta_T,
		'centroid_indices': cent_T_idx,
		'psi_mun_fresh':    psi_mun_fresh_T,
		'psi_nmu_fresh':    psi_nmu_fresh_T,
	}


def resolve_zeta_fit_edge(band_slices, zeta_nband):
	"""The ζ-fit edge this run actually gets: an int that NARROWS, or ``None``.

	**One resolver, three consumers, one banner.**  ``zeta_nband`` is a
	LOGICAL band count written in a deck; the edge the fit is handed is
	``band_slices.b4``, that count rounded up to the world size.  Those two
	numbers differ on every deck whose ``nband`` does not divide the process
	count, and until 2026-08-22 the difference was resolved in the parser,
	which cannot see ``b4``:

	    P=4, deck ``nband = zeta_nband = 14``  ->  parser erased the key as
	    "redundant"  ->  fit ran on [0,16)  ->  refused, because band 16 cuts
	    a multiplet.  The deck had asked for 14.
	    (JID 57152792, ``runs/Si_scalar/11_scalar_v_rootcause_20260817/``)

	So the comparison is made HERE, against the padded edge:

	* ``None``               -> follow the loaded window; ``b4`` passes through
	  untouched and the fit is bit-identical to every pre-2026-08-11 deck.
	* ``== band_slices.b4``  -> the request IS the loaded window.  Nothing to
	  narrow, so it collapses to the same "follow" path — this is the case the
	  parser used to catch, and it still behaves identically whenever the
	  deck's ``nband`` divides the world size.
	* anything else          -> an explicit narrowing, honoured exactly.  Note
	  this includes ``zeta_nband == nband < b4``: the deck named an edge, and
	  :func:`zeta_fit_band_ranges` does not re-apply the pad to a requested
	  edge ("honoured exactly or refused" — see its own docstring).

	Pure and jax-free, so it is testable without a mesh; it takes
	``band_slices`` rather than a config because ``b4`` is the only thing it
	is allowed to compare against.
	"""
	if zeta_nband is None:
		return None
	edge = int(zeta_nband)
	if edge == int(band_slices.b4):
		return None
	return edge


def zeta_fit_band_ranges(band_slices, zeta_nband, *, log=print):
	"""The two band ranges the ISDF ζ fit runs on: ``(left, right)``.

	``left = (b0, b3)`` ("all val + sigma cond") and ``right = (b1, b4)``
	("sigma val + all cond") — the pair density needs asymmetric ranges — for
	every deck written before 2026-08-11, and for every deck since that does
	not name ``zeta_nband``.  ``zeta_nband is None`` returns exactly those two
	tuples and nothing else happens here, which is what makes the default
	bit-identical: ``b4`` is the PADDED edge and is passed through untouched.

	THE DECOUPLING.  ``b4`` is the top of the χ0/Σ band sum AND, until today,
	the top of the window ζ was fitted on.  The two want opposite things.  The
	fit wants a NARROW window: the per-Q ζ refit ``bse.exciton_bands`` runs for
	a dense exciton band path reaches its target Q through an htransform
	Galerkin representation whose rank bound is ``n_μ·n_s ≥ nk·nb``, and on the
	Si 4×4×4 / 2628-centroid parent the measured capacity point is nb ≈ 52
	(``build_fH_R``'s orthonormality gate reads 3.44e-07 at nb 52 against a
	1.0e-06 cap, and 3.47e-06 at nb 60).  The band sum wants a WIDE one.  With
	one key for both, narrowing ``nband`` to serve the fit also truncated the
	sum — ``BandSlices`` requires b3 ≤ b4, so ``ncond`` had to come down with
	it — and moved every quasiparticle level by ~222 meV median over the 4v8c
	window (48 meV in the direct gap) for reasons that are not about the ζ
	basis at all (``tests/known_failures/2026-08-11-narrowed-zeta-window-\
clears-fh-and-the-tile-null-still-refuses.md`` §3).

	``zeta_nband`` narrows ONLY the fit; the caller's ``band_slices`` — and so
	χ0, Σ and the restart bundle's band axis — are untouched.

	THE PAD IS DELIBERATELY NOT RE-APPLIED.  ``b4`` is
	``round_up(nband, world_size)`` because the band axis has to divide the
	device mesh for the sharded readers that FILL it.  This edge only SLICES an
	array that already exists, and rounding it up would move a
	degeneracy-checked edge to a different, unchecked one whenever the world
	size changed — a physical band window that depends on the process count is
	exactly what ``Meta``'s own pad note is careful to keep out of the physics.
	A requested edge is honoured exactly or refused.

	THE SIZING RULE, STATED RATHER THAN INHERITED (2026-08-16, the χ/Σ split).
	The fit's right range tops out at ``b4``, and ``b4`` is the padded top of
	``max(number_bands_chi, number_bands_sigma)``.  That ``max`` is not an
	implementation convenience — it is the guard against a documented
	eV-scale failure.  ``docs/dev/isdf_basis_adequacy_at_large_nband.md``
	records a run whose ISDF window was clamped to a SMALL band range and used
	for a large one: QP gap **0.36 eV where the answer is 3.1–3.7 eV**, with a
	NEGATIVE ``eqp1``, passing every gate in the suite (el_compare 1.9e-11, H0
	3.9e-5, W Dyson 1.9e-14, Σ_X within 0.03 %) because every one of them is
	upstream of or orthogonal to Σ_c.  Re-selecting the SAME NUMBER of
	centroids against a representative window moved the gap monotonically to
	3.14 / 3.72 eV.

	So: the interpolation basis must span the pair densities of whichever
	consumer reaches higher.  Sizing it by the SMALLER count would rebuild
	exactly that failure, one index over — χ0 or Σ (whichever is the larger)
	would consume pair densities the basis was never fitted to represent, and
	nothing downstream would notice.  :func:`assert_isdf_window_is_the_max`
	states the invariant where it can fail; this is why.
	"""
	left = (band_slices.b0, band_slices.b3)
	right = (band_slices.b1, band_slices.b4)
	if zeta_nband is None:
		return left, right
	b4_zeta = int(zeta_nband)
	if not (band_slices.b1 < b4_zeta <= band_slices.b4):
		raise ValueError(
			f"zeta_nband={b4_zeta} is outside the band window this run holds: "
			f"the ζ fit's right range starts at b1={band_slices.b1} and the "
			f"centroid ψ spans [b0, b4) = [{band_slices.b0}, "
			f"{band_slices.b4}).  zeta_nband can only NARROW the ζ-fit "
			f"window; it cannot move it outside the loaded bands.")
	left = (band_slices.b0, min(band_slices.b3, b4_zeta))
	right = (band_slices.b1, b4_zeta)
	log(f"    ζ-fit window DECOUPLED from the band sum: zeta_nband="
	    f"{b4_zeta} (nband/b4={band_slices.b4}).  χ0/Σ still sum "
	    f"[{band_slices.b0}, {band_slices.b4}); ζ is fitted on left {left} "
	    f"x right {right}.")
	if band_slices.b3 > b4_zeta:
		log(f"    *** zeta_nband={b4_zeta} is BELOW the Σ evaluation window's "
		    f"top b3={band_slices.b3}.  Quasiparticle energies for bands "
		    f"[{b4_zeta}, {band_slices.b3}) are then built on pair densities "
		    f"whose bra leg was never fitted — the ζ basis is EXTRAPOLATED "
		    f"there.  Lower ncond to {b4_zeta - band_slices.b2} if those "
		    f"bands are wanted. ***")
	return left, right


def check_zeta_fit_windows(energies, band_range_left, band_range_right,
                           zeta_nband, logical_band_stop, *, log=print):
	"""ARE THESE TWO WINDOWS POINT-GROUP-INVARIANT SUBSPACES?

	The guard already exists — ``common.band_degeneracy`` — and the BSE has
	called it since 2026-08-10.  The ζ fit never did, and this is the seam
	where it matters most: ζ is what the IBZ cascade unfolds, so a window that
	is not invariant here breaks the k-star identity for EVERY object built on
	ζ, W included, and Σ_x first of all.

	Why a rotation is the reason: the cascade builds the full BZ by rotating
	the wedge, and a rotation sends ψ_n(k) into a combination of its
	DEGENERATE PARTNERS at Sk.  A window containing half a multiplet therefore
	has a rotation image that leaves it, and the pair space it represents is
	not invariant.  This is ``common/rank_criterion``'s story one index over —
	except that a band degeneracy is EXACT, so unlike a spectral cut there is
	no tolerance to tune.

	MEASURED, Si 6×6×6, on the stock windows (nval 8 / ncond 52 / nband 60 →
	left [0,60), right [0,60)); the top edge cuts a 4-fold manifold (bands
	59..62) at 4 of the 16 wedge k, keeping 2 and dropping 2:

	    nband=60 (open)     Σ_x star spread 0.0640 meV   Σ_c 38.785 meV
	    nband=68 (closed)   Σ_x star spread 0.0000 meV   Σ_c  0.083 meV

	and λ_max(C_q) — an exact star invariant — goes from star-constant only to
	1e-4 to star-constant to 1e-10, which is the 4×4×4 anchor's own level.  The
	ζ rank truncation still fires on all 216 q in the closed arm, which is what
	rules the truncation out as the cause.
	``tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md``

	STRICT AT THE CONSUMER.  A cut edge is not made safer by having arrived
	through an old ``nband``/``ncond`` key.  Scalar-Si c192 gave the missing
	discriminator on 2026-08-17: the same WFN and centroids at fit edge 12 had
	``V(q)-conj(V(-q)) = 9.691e-2``, while moving only the edge to the closed
	band 14 gave ``2.685e-8``.  The wavefunction prefix itself failed TRS
	subspace closure by 3.79e-1 at edge 12 and passed at 3.50e-9 at edge 14;
	the full degenerate blocks passed at 4.86e-9.  On that controlled pair the
	fit window is the first broken object.  This does not claim that every
	reciprocity residual has that one cause: the 648-centroid production fit
	retains a separate q-unfold spatial-realization defect even at a closed
	window, handled at the q-grid owning site rather than weakened here.
	Every fit edge therefore uses ``strict`` and refuses before the expensive
	fit.  This matches :mod:`common.band_degeneracy`'s owner-selected default;
	the old warning-only grandfather clause is removed.

	A fit ending at the LAST band stored in WFN.h5 is also refused.  The
	outer-edge ``+inf`` convention only means "nothing beyond this ARRAY"; it
	does not prove the physical multiplet ended there.  The scalar-Si 60-band
	production WFN demonstrates the distinction: its occupied density passed
	all 48 spatial operations at 5.43e-10, yet the selected 60-band prefix had
	wavefunction-level little-group residual 9.96e-1 and produced V reciprocity
	3.280e-2.  A WFN
	with at least one spare band makes the top gap measurable.  Without that
	spare, closure is absent evidence, not a pass.

	The fit ranges themselves retain mesh padding, but pad bands are exact-zero
	storage slots, not a physical subspace.  ``logical_band_stop`` is therefore
	the upper edge certified here; it is ``zeta_nband`` when explicitly narrowed
	and otherwise the unpadded maximum band count held by ``Meta``.
	"""
	if energies is None:
		log("  [band window] closure NOT CHECKED: this loader exposes no "
		    "`energies`.  That is an absence, not a pass.")
		return
	from common import band_degeneracy as _bd
	enk = np.asarray(energies)[0]
	# `enk` is `energies[0]` -- the loader's WHOLE ladder, not the zeta
	# window -- which is what lets this seam see that the window's own
	# edge slices.  Handing it the window instead would report +inf and
	# certify the cut (band_degeneracy.boundary_min_gaps, 2026-08-15).
	gaps = _bd.boundary_min_gaps(enk, is_full_spectrum=True)
	def _physical_hi(hi_storage):
		hi_storage = int(hi_storage)
		if (hi_storage == int(band_range_right[1])
				and hi_storage > int(logical_band_stop)):
			return int(logical_band_stop)
		return hi_storage
	for lo, hi_storage, what in (
			(band_range_left[0], band_range_left[1], "ISDF left window"),
			(band_range_right[0], band_range_right[1], "ISDF right window")):
		hi = _physical_hi(hi_storage)
		if int(hi) >= int(enk.shape[1]):
			raise _bd.BandWindowDegeneracyError(
				f"[band-window] {what} (the ζ fit's pair space): upper "
				f"boundary band {int(hi)} reaches the available WFN extent "
				f"({int(enk.shape[1])} bands), so its degeneracy closure cannot "
				f"be checked.  The outer-array boundary cuts nothing in the "
				f"stored table but may still cut a physical multiplet.  Regenerate "
				f"WFN.h5 with at least one spare band above {int(hi)}, then rerun; "
				f"a fit window is accepted only when its upper gap is measured.")
		_bd.check_band_window(
			enk, int(lo), int(hi), mode="strict",
			log=log,
			where=(f"{what} (the ζ fit's pair space)"
			       + (f" — deck key zeta_nband={zeta_nband}"
			          if zeta_nband is not None and int(hi) == int(zeta_nband)
			          else "")))
	# Print the number even when it is fine: "no news" and "a good number"
	# must not look alike (preamble measurement rule 10).
	edges = sorted({_physical_hi(band_range_left[1]),
	                int(band_range_right[0]),
	                _physical_hi(band_range_right[1])})
	log("    ζ band-window closure: " + ", ".join(
		f"edge {b} min gap {gaps[b] * 13605.693122994:.3g} meV"
		if b < len(gaps) and np.isfinite(gaps[b])
		else f"edge {b} exempt (cuts nothing)"
		for b in edges))


#: Env override for the band-sum degeneracy guard: ``snap`` (or ``off``)
#: downgrades the refusal below.  Same three-mode vocabulary as
#: ``common.band_degeneracy.MODES`` and the BSE's ``--band-degeneracy``; it is
#: an ENV knob rather than a deck key because it is an escape hatch for a
#: deck you are debugging, not a property of the calculation you would want
#: recorded in the input file.  ``AGENT_PREAMBLE``: never set it to make a
#: gate pass.
_BAND_DEGENERACY_ENV = "LORRAX_BAND_DEGENERACY"


def assert_isdf_window_is_the_max(band_slices, band_range_right, zeta_nband,
                                  *, log=print):
	"""THE SIZING RULE, ENFORCED AT THE SEAM THAT CONSUMES IT.

	*The ISDF basis must span the band window its consumers actually reach.*
	With one band count that was automatic.  With two it is a ``max``, and a
	``max`` that is only implied by how ``b4`` happens to be computed is one
	refactor away from being the ``min``.

	WHAT IT COSTS TO GET WRONG, measured, not feared.
	``docs/dev/isdf_basis_adequacy_at_large_nband.md``: a run whose ISDF
	window was clamped to a small band range and then used for a large one
	returned a QP gap of **0.36 eV** where the answer is **3.1–3.7 eV**, with
	a NEGATIVE ``eqp1`` fundamental gap — and **passed every gate the project
	runs**, because all of them are upstream of or orthogonal to Σ_c
	(el_compare 1.86e-11 eV, gate_h0 3.9e-5 eV, W Dyson residual 1.9e-14,
	density/TRS 1.3e-14, bare Σ_X within 0.03 %, head fit 0.026 %).  The fix
	was to re-select the SAME NUMBER of centroids against a representative
	window; the gap then moved monotonically to 3.135 / 3.723 eV.  So this is
	a class of error that is invisible to everything except a check placed
	exactly here.

	THREE THINGS ARE CHECKED, and they are different questions:

	1. The fit's right range tops out at ``max(b4_chi, b4_sigma)`` (== ``b4``,
	   the padded edge).  A window sized by the smaller consumer is refused
	   outright — that is the failure above, one index over.
	2. If ``zeta_nband`` narrows the fit BELOW either band sum's top, the
	   consumer above it is running on an EXTRAPOLATED ζ basis.  Reported per
	   consumer, loudly, and named — narrowing is a legitimate request
	   (``zeta_nband`` exists for the BSE's Galerkin capacity bound) but it
	   must never be silent about which sum it undercuts.
	3. The window and the count that set it are logged unconditionally, so
	   "no news" and "a good number" do not look alike.
	"""
	top = int(band_range_right[1])
	b4_chi, b4_sigma = int(band_slices.b4_chi), int(band_slices.b4_sigma)
	expected = max(b4_chi, b4_sigma)
	if zeta_nband is None and top != expected:
		raise ValueError(
			f"ISDF ζ-fit window top is {top} but the band sums reach "
			f"max(chi={b4_chi}, sigma={b4_sigma}) = {expected}.  The "
			f"interpolation basis MUST span the pair densities of whichever "
			f"consumer reaches higher; a basis fitted to the smaller window "
			f"and used for the larger one is the 0.36 eV / negative-gap "
			f"failure in docs/dev/isdf_basis_adequacy_at_large_nband.md, "
			f"which passed every gate in the suite.  This is a code defect, "
			f"not a deck error.")
	source = ("tied" if b4_chi == b4_sigma
	          else ("number_bands_chi" if b4_chi > b4_sigma
	                else "number_bands_sigma"))
	log(f"    ζ-fit window sized for {top} bands "
	    f"(band sums reach chi {b4_chi}, sigma {b4_sigma}; the max is set by "
	    f"{source})"
	    + ("" if zeta_nband is None
	       else f", NARROWED from {expected} by deck key zeta_nband={top}"))
	if zeta_nband is not None:
		for edge, what, key in ((b4_chi, "χ0/W band sum", "number_bands_chi"),
		                        (b4_sigma, "Σ band sum", "number_bands_sigma")):
			if top < edge:
				log(f"    *** zeta_nband={top} is BELOW the {what}'s top "
				    f"({key} → band {edge}).  Bands [{top}, {edge}) of that "
				    f"sum are built on pair densities whose ζ basis was "
				    f"never fitted to them — the basis is EXTRAPOLATED "
				    f"there.  This is the mechanism behind the 0.36 eV gap "
				    f"in docs/dev/isdf_basis_adequacy_at_large_nband.md; it "
				    f"is permitted because you asked for it by name, and it "
				    f"is not checked by any other gate. ***")


def check_band_sum_degeneracy(wfn, cfg, band_slices, *, log=print):
	"""Do the χ and Σ band-sum tops each cut a clean multiplet boundary?

	SAME QUESTION AS ``check_zeta_fit_windows``, ONE INDEX OVER, and the same
	answer for the same reason: a band sum truncated inside a degenerate
	multiplet keeps half an irrep, and half a multiplet is not a subspace of
	anything.  Measured on the Si 4×4×4 SOC deck, ``nband = 60`` slices a
	multiplet and costs **1.957 meV of within-star Σ spread** — a quantity
	that is exactly zero when the cut is clean (floor 0.0010 meV).  Star
	covariance is an identity, not a convergence parameter, so that spread is
	an error with no knob to reduce it.

	MODE, AND WHY IT IS NOT UNIFORM — the same grandfather clause
	``check_zeta_fit_windows`` documents, and for the same census reason:

	* An edge that arrives through the UMBRELLA (``number_bands`` / the
	  transitional ``nband``) is checked ``snap`` — reported loudly, never
	  fatal.  Every deck in the tree sits on such an edge, chosen before this
	  check existed; flipping them to ``strict`` would refuse decks that have
	  been producing frozen references for months, and it would also break
	  the bit-identity claim this feature is required to hold (a refusal is
	  not a byte-identical ``eqp0.dat``).
	* An edge the deck NAMED — ``number_bands_chi`` or
	  ``number_bands_sigma`` — REFUSES.  Naming one of these keys is a
	  brand-new, explicit request for a specific edge on a deck that by
	  construction has no history, which is precisely the argument that made
	  ``zeta_nband``'s edge strict on 2026-08-11.

	``LORRAX_BAND_DEGENERACY=snap`` (or ``off``) downgrades the refusal for
	both, and says so in the log when it does.  The two counts are checked
	INDEPENDENTLY and the message names the legal edges for the one that
	failed, so a deck with two bad edges learns about both.
	"""
	from common import band_degeneracy as _bd
	energies = getattr(wfn, "energies", None)
	if energies is None:
		log("  [band sum] closure NOT CHECKED: this loader exposes no "
		    "`energies`.  That is an absence, not a pass.")
		return
	enk = np.asarray(energies)[0]
	nb = int(enk.shape[1])
	# ``enk`` is ``wfn.energies``, the UNTRUNCATED mean-field ladder -- not
	# the Sigma window.  That is exactly the pattern b27f98c3's docstring
	# prescribes ("pass the untruncated ladder and the window bounds"):
	# the edges checked below (b4_chi, b4_sigma) are WINDOW bounds, and the
	# only array that can say whether they slice a multiplet is the full
	# ladder.  Same array and same answer as gw_init.py's other call site.
	gaps = _bd.boundary_min_gaps(enk, is_full_spectrum=True)
	override = os.environ.get(_BAND_DEGENERACY_ENV, "").strip().lower()
	if override and override not in _bd.MODES:
		raise ValueError(
			f"{_BAND_DEGENERACY_ENV}={override!r} is not one of "
			f"{_bd.MODES}.  A guard override with an unrecognised value is "
			f"not silently ignored.")
	for edge, key, what in (
			(int(band_slices.b4_chi), "number_bands_chi", "chi0/W band sum"),
			(int(band_slices.b4_sigma), "number_bands_sigma",
			 "Sigma band sum")):
		named = key in cfg.bands.named
		mode = override or ("strict" if named else "snap")
		# The PADDED edge cuts nothing real: ψ above b_id_4_user is zero, so
		# the boundary there is between a real band and a pad band and the
		# gap is meaningless.  Ask about the LOGICAL count instead, which is
		# what the physics truncates at.
		logical = min(edge, int(cfg.bands.chi if key.endswith("chi")
		                        else cfg.bands.sigma))
		if logical >= nb:
			detail = (
				f"{what} is truncated at {logical} bands, exactly at/past the "
				f"{nb}-band WFN extent.  Multiplet closure is NOT CHECKABLE: "
				"the next physical band is absent, so this is absence of evidence, "
				"not a clean boundary.  Supply a WFN with at least one spare band "
				"above this sum edge to certify it.")
			if mode == "strict":
				raise _bd.BandWindowDegeneracyError(
					f"{detail}  This edge was requested BY NAME (`{key}` = "
					f"{logical}), so unverifiable closure is refused.  Increase the "
					f"WFN band count, choose a lower certified `{key}`, or set "
					f"{_BAND_DEGENERACY_ENV}=snap to continue diagnostically.")
			if mode == "off":
				log(f"    *** {detail}  ({_BAND_DEGENERACY_ENV}=off: NOT "
				    "CHECKED.) ***")
			else:
				log(f"    *** {detail}  Continuing because this legacy umbrella "
				    "edge is grandfathered to a warning"
				    + (f" ({_BAND_DEGENERACY_ENV}={override} was set)."
				       if override else ".") + " ***")
			continue
		gap_mev = float(gaps[logical]) * 13605.693122994
		if gap_mev > _bd.DEGENERACY_TOL_RY * 13605.693122994:
			log(f"    {what} edge {logical} clean: min gap "
			    f"{gap_mev:.3g} meV.")
			continue
		legal = [b for b in range(int(band_slices.b2) + 1, nb)
		         if gaps[b] > _bd.DEGENERACY_TOL_RY]
		near = ([b for b in legal if b <= logical][-3:]
		        + [b for b in legal if b > logical][:3])
		detail = (
			f"{what} is truncated at {logical} bands, and that boundary "
			f"SPLITS A DEGENERATE MULTIPLET: the tightest gap across it "
			f"anywhere in the BZ is {gap_mev:.4g} meV, against a "
			f"{_bd.DEGENERACY_TOL_RY * 13605.693122994:.3g} meV tolerance.  "
			f"Half a multiplet is not a symmetry-adapted subspace; on the Si "
			f"4x4x4 SOC deck a sliced band sum cost 1.957 meV of within-star "
			f"Sigma spread, which is an identity violation and has no "
			f"convergence knob.  Legal edges for THIS count near {logical}: "
			f"{near or 'none in range'}.")
		if mode == "off":
			log(f"    *** {detail}  ({_BAND_DEGENERACY_ENV}=off: NOT "
			    f"CHECKED.) ***")
		elif mode == "snap":
			# THROUGH ``sanity.warn``, not a bare log line.  This branch is
			# the one that measures a defect and PROCEEDS, and until now its
			# output carried none of the markers the rest of the tree's
			# failure signatures use: a log grep for `*** LORRAX SANITY
			# FAILURE` -- the token every other gate emits and every log
			# triage grep looks for -- came back clean on a run whose band
			# sum was measurably sliced.  MEASURED consequence on the Si
			# 4x4x4 SOC anchor: within-star Sigma spread 1.957 meV at
			# nband=60 against exactly 0.0000 meV at the degeneracy-clean
			# nband=40 and 36, same centroids and same zeta_rcond, only the
			# edge moving.  ``LORRAX_SANITY=strict`` now stops the run here,
			# which is what a regression gate wants; the default still
			# continues, because flipping the umbrella grandfather clause to
			# a refusal would refuse decks that have been producing frozen
			# references for months and is the OWNER's call, not this
			# guard's.
			from common import sanity
			sanity.warn(
				f"{detail}  Continuing: "
				+ (f"{_BAND_DEGENERACY_ENV}={override} was set."
				   if override else
				   f"this edge arrived through the umbrella `number_bands`, "
				   f"which is grandfathered to a warning.  Set `{key}` "
				   f"explicitly to have it enforced."),
				print_fn=log)
		else:
			raise _bd.BandWindowDegeneracyError(
				f"{detail}  This edge was requested BY NAME (`{key}` = "
				f"{logical}), so it is enforced rather than warned about.  "
				f"Set `{key}` to one of the legal edges above, or set "
				f"{_BAND_DEGENERACY_ENV}=snap to continue anyway (it logs "
				f"loudly, and it changes the physics rather than fixing it).")


def _plan_gflat_chunks_for_channel(
		*, meta, cfg, band_slices, mesh_xy, is_bispinor, print_fn=print):
	"""Chunk-plan ONE ISDF centroid channel: the charge channel
	(``meta.n_rmu``) or one transverse channel (``meta.n_rmu`` — μ_T is
	typically ≈ μ_C/3).

	:func:`gw.gflat_memory_model.plan_gflat_chunks` is already
	channel-agnostic — μ comes entirely from ``meta.n_rmu_padded`` /
	``meta.n_rmu`` — but until this function existed only ONE call site
	ever invoked it (``prepare_isdf_and_wavefunctions``, charge-only), so
	all three transverse ζ_T fits inherited that CHARGE-sized ``chunks``
	dict wholesale (register: "three ζ_T fits inherit the CHARGE chunk
	plan (μ_T≈μ_C/3): ~3x extra r-chunks, ~2.7 GB/rank avoidable
	gather").  This function is the ONE place that resolves
	``plan_gflat_chunks``'s other inputs (``nb_total``, the fit-window
	union, ``ngkmax``, the infeasibility refusal) so the charge and
	transverse call sites cannot drift apart — "one planner, channel-
	parameterized," not a second plan.

	Returns ``(chunks, gflat_plan)`` — ``chunks`` is the plain dict
	``fit_zeta`` / ``fit_zeta_to_h5`` consume (``band_chunk`` /
	``chunk_r`` / ``q_chunk`` / ``gflat_chunk_size`` /
	``memory_estimate``); ``gflat_plan`` is the raw
	:class:`gw.gflat_memory_model.GFlatChunkPlan` for a caller that wants
	more than the dict exposes.
	"""
	from gw.gflat_memory_model import plan_gflat_chunks
	mem = cfg.memory
	nb_total = ((band_slices.b3 - band_slices.b0)
	            + (band_slices.b4 - band_slices.b1))
	# The resident centroid copies follow the full consumer band
	# inventory above, while the pair-GEMM K dimension follows the
	# (possibly zeta_nband-narrowed) union of the ζ fit windows.
	# Resolve the latter without logging; fit_zeta owns the one
	# user-facing window announcement and its degeneracy checks.
	_zeta_left, _zeta_right = zeta_fit_band_ranges(
		band_slices,
		resolve_zeta_fit_edge(band_slices, getattr(cfg, "zeta_nband", None)),
		log=lambda _message: None)
	_zeta_fit_nb = (max(_zeta_left[1], _zeta_right[1])
	                - min(_zeta_left[0], _zeta_right[0]))
	# Q-axis on disk: conservative full-BZ (the transverse path writes
	# IBZ-only, which is smaller).
	_ngkmax = int(getattr(meta, 'ngkmax', 0)) or int(0.06 * meta.n_rtot)
	gflat_plan = plan_gflat_chunks(
		meta=meta, mesh_xy=mesh_xy,
		nb_total=nb_total, fit_nb_total=_zeta_fit_nb,
		ngkmax=_ngkmax,
		n_q_disk=int(meta.nk_tot),
		budget_gb=float(mem.per_device_gb),
		target_utilization=(mem.chunk_target_utilization
		                    if mem.chunk_target_utilization > 0 else None),
		is_bispinor=bool(is_bispinor),
		max_chunks=64,
		r_chunk_override=(int(mem.r_chunk_override)
		                  if mem.r_chunk_override > 0 else None),
		band_chunk_override=(int(mem.band_chunk_size)
		                     if mem.band_chunk_size > 0 else None),
		gflat_chunk_size_override=(int(mem.gflat_chunk_size)
		                           if mem.gflat_chunk_size > 0 else None),
		distributed_zeta_solve=str(cfg.backend.distributed_zeta_solve),
		low_mem_bands=bool(mem.low_mem_bands),
		# Stage F writes per-rank hyperslabs; the planner therefore
		# charges only the local sharded tile.
	)
	if jax.process_index() == 0:
		print_fn("")
		print_fn(gflat_plan.format())
	# A plan the planner itself prices as infeasible is a REFUSAL, not a
	# warning.  The 9x9/626b run printed "244% of budget (expect OOM)"
	# here, proceeded, and OOMed at the first z_q_phase (JID 57281385
	# step .28) — the instrument-that-measures-and-proceeds class.  An
	# explicit ``r_chunk_size`` keeps its documented run-level-workaround
	# authority: with it set, the operator has asserted the chunking and
	# only the warning prints.
	_over = (gflat_plan.hwm_bytes > gflat_plan.budget_bytes)
	_floor_broken = gflat_plan.p_min > mesh_xy.devices.size
	if _floor_broken or _over:
		_msg = (
			f"[planner] the certified plan does not fit: "
			f"HWM {gflat_plan.hwm_bytes / 1e9:.2f} GB/dev vs budget "
			f"{gflat_plan.budget_bytes / 1e9:.2f} GB/dev "
			f"(binder: {gflat_plan.bottleneck})"
			+ (f"; rank floor P_min={gflat_plan.p_min} exceeds the "
			   f"{mesh_xy.devices.size} ranks in this mesh — no "
			   f"chunk choice can shrink the persistent ÷P floor"
			   if _floor_broken else "")
			+ ".  Add ranks or raise memory_per_device_gb"
			# NAME ONLY THE REMEDIES THAT ACTUALLY APPLY.  An explicit
			# ``r_chunk_size`` bypasses this gate for a budget overrun
			# and does NOTHING for a broken rank floor (the floor is
			# the persistent ÷P term, which no chunk choice touches) —
			# and the operator who is already running with one would
			# be told to set the thing they set.  A refusal that names
			# an inapplicable fix is a dead end wearing a remedy's
			# clothes.
			+ ("" if (_floor_broken or mem.r_chunk_override > 0)
			   else ", or set an explicit r_chunk_size — the "
			        "register-documented run-level workaround, "
			        "which bypasses this gate")
			+ ".")
		if mem.r_chunk_override > 0 and not _floor_broken:
			print_fn(f"  {_msg}  Proceeding under the explicit "
			       f"r_chunk_size={int(mem.r_chunk_override)} the "
			       f"operator asserted; the plan is still priced "
			       f"over budget.")
		else:
			raise ValueError(_msg)
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
	return chunks, gflat_plan


def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
             psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print,
             psi_nmu_fresh=None, psi_mun_fresh=None):
	"""Fit ISDF interpolation vectors ζ and write to HDF5.

	The caller supplies (a) the full-range centroid wavefunctions
	(``psi_rmu_Y`` / ``psi_rmuT_X``, spanning [b0, b4) as returned by
	``load_centroids_band_chunked``) and (b) the chunk plan dict from
	:func:`gw.gflat_memory_model.plan_gflat_chunks`.  Returns
	``(zeta_h5_path, mem_est, transverse_wfn_data)``.

	``psi_nmu_fresh``/``psi_mun_fresh``: the two-face carrier, required
	(and BOTH ``psi_rmu_Y`` and ``psi_rmuT_X`` expected ``None``) when
	``cfg.memory.low_mem_bands`` — see ``prepare_isdf_and_wavefunctions``,
	which builds them right after the fresh load and drops both
	single-axis copies before calling here (neither has a consumer left:
	the CCT Gram build and the r-chunk loop's band contraction both read
	the face carrier — see ``isdf.core._c_q_face``/``_z_q_face`` and
	docs/architecture/zeta_fit_face_psi_cct.md).  Forwarded to the
	charge-channel ``fit_zeta_to_h5`` call below.  The bispinor
	transverse-channel calls build their OWN face carrier
	(``psi_mun_fresh_T``/``psi_nmu_fresh_T``, from the transverse
	centroid load, same ``PSI_MUN_SPEC``/``PSI_NMU_SPEC`` build path —
	2026-08-23) rather than reusing this function's own
	``psi_nmu_fresh``/``psi_mun_fresh`` parameters, since the two
	centroid sets (charge μ, transverse μ_T) are different arrays.
	"""
	from gw.isdf_fitting import fit_zeta_to_h5
	from common.gamma_matrices import set_gamma_contract_mode
	# Honour cohsex.in ``gamma_contract_mode`` for the γ̃·γ̃ kernel
	# inside the monolithic pair pipeline.  Mode is module-level (the
	# γ̃ contract sits inside shard_map bodies so threading a kwarg
	# through every call would be churn for no benefit).
	set_gamma_contract_mode(cfg.backend.gamma_contract_mode)

	# ISDF left/right band windows (pair density needs asymmetric ranges),
	# and the ``zeta_nband`` decoupling if the deck asked for one.
	# ONE resolved edge, three consumers.  ``cfg.zeta_nband`` is the deck's
	# logical request; ``resolve_zeta_fit_edge`` compares it against the
	# PADDED ``band_slices.b4`` and returns None when there is nothing to
	# narrow.  Reading ``cfg.zeta_nband`` directly below would let the three
	# gates disagree about whether this run is decoupled at all.
	zeta_edge = resolve_zeta_fit_edge(
		band_slices, getattr(cfg, "zeta_nband", None))
	band_range_left, band_range_right = zeta_fit_band_ranges(
		band_slices, zeta_edge, log=print_fn)
	assert_isdf_window_is_the_max(
		band_slices, band_range_right, zeta_edge, log=print_fn)
	check_zeta_fit_windows(
		getattr(wfn, "energies", None), band_range_left, band_range_right,
		zeta_edge,
		(int(zeta_edge) if zeta_edge is not None
		 else int(getattr(meta, "b_id_4_user", 0) or band_slices.b4)),
		log=print_fn)

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
	# which run AFTER the charge fit completes, so an explicit
	# distributed_lu whose transverse centroid count does not divide the
	# mesh burned the multi-hour charge fit before the inevitable
	# ValueError.  Every input needed to refuse — mesh shape,
	# distributed_lu, the transverse centroid count — is known right here
	# (one text-file read), so a doomed run refuses in seconds; the same
	# call announces an ``auto`` demotion up front (rank 0).  The
	# per-channel resolution inside fit_zeta_to_h5 later re-resolves
	# identically (defense in depth).
	#
	# The pre-flight also OWNS the transverse channel's identity for the
	# rest of this function (2026-08-04): the centroid table, its μ
	# extent, ``meta_T`` and the resolved solver kind are all needed
	# before the ζ-reuse decision — which now covers the transverse
	# channel — and were previously rebuilt from scratch after the charge
	# fit.  One construction site, consumed by the provenance stamp, the
	# reuse check, the ψ sampling and the μ_L fit loop alike.
	_meta_T = None
	_cent_T_idx = None
	_transverse_identity = None
	_chunks_T = None
	# Transverse ζ IBZ-write activates whenever the bispinor V_q
	# orchestrator iterates IBZ q's — the SAME gate the charge ζ uses,
	# so it is derived from the charge value rather than re-reading the
	# environment (one env_bool call, one announcement).
	_write_ibz_only_transverse = bool(cfg.bispinor) and _write_ibz_only_charge
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
		import dataclasses
		from file_io.centroids import load_centroids as _load_cent_pf
		from isdf.core import _resolve_solver_kind_transverse
		from runtime.padding import padded_mu_extent
		_, _cent_T_idx_np, _n_rmu_T_pf = _load_cent_pf(
			cfg.paths.centroids_file_current, meta.fft_grid)
		_transverse_solver_kind = _resolve_solver_kind_transverse(
			mesh_xy, cfg.backend.distributed_lu,
			n_rmu_logical=int(_n_rmu_T_pf),
			transverse_zeta_solve=cfg.backend.transverse_zeta_solve)
		_cent_T_idx = jnp.asarray(_cent_T_idx_np, dtype=jnp.int32)
		# n_rmu_padded uses world_size (= ∏ p_a over the device mesh).
		# Without this refresh the bispinor transverse fit_zeta inherits
		# the charge-channel padded extent, and the C_q reshape at
		# isdf_fitting.py:1442 trips a TypeError when the transverse
		# centroid count differs from the charge count.
		# ``padded_mu_extent`` also honors the test-only
		# LORRAX_EXTRA_MU_PAD knob (pad-extent-invariance gate).
		_meta_T = dataclasses.replace(
			meta,
			n_rmu=int(_n_rmu_T_pf),
			n_rmu_padded=int(padded_mu_extent(int(_n_rmu_T_pf),
			                                  int(jax.device_count()))),
		)
		# ``sys_dim`` is set dynamically on ``meta`` by gw_jax.main
		# (Meta has no sys_dim field), so dataclasses.replace doesn't
		# carry it over.  Copy it explicitly — fit_zeta_to_h5 reads
		# meta.sys_dim when building the per-q G-flat sphere.
		_meta_T.sys_dim = meta.sys_dim
		# Hashed off the HOST table (no device round-trip), which is the
		# same int64 view ``_centroid_table_md5`` takes of the device
		# array stamped on the restart tensors — the two hashes of one
		# centroid file are equal by construction.
		_transverse_identity = {
			'n_rmu':          int(_n_rmu_T_pf),
			'centroids_md5':  _centroid_table_md5(_cent_T_idx_np),
			'distributed_lu': str(cfg.backend.distributed_lu).strip().lower(),
			'solver_kind':    str(_transverse_solver_kind),
		}
		# Chunk-plan the TRANSVERSE channel SEPARATELY from the charge
		# ``chunks`` this function was handed.  μ_T is typically ≈ μ_C/3,
		# and reusing the charge-sized plan unchanged for all three ζ_T
		# fits is exactly the register row this closes: "three ζ_T fits
		# inherit the CHARGE chunk plan (μ_T≈μ_C/3): ~3x extra r-chunks,
		# ~2.7 GB/rank avoidable gather".  ONE call here, ahead of both
		# the ζ-reuse decision and the μ_L loop, so the ψ sampling (fit
		# AND reuse paths) and all three Lorentz components — which share
		# one transverse centroid set — share one transverse-sized plan,
		# exactly mirroring how the charge channel gets one plan.
		_chunks_T, _ = _plan_gflat_chunks_for_channel(
			meta=_meta_T, cfg=cfg, band_slices=band_slices, mesh_xy=mesh_xy,
			is_bispinor=True, print_fn=print_fn)
	# ── ζ REUSE: skip the fit when the ζ files are complete AND provably
	# the same fit.  Before this, a rerun in the same directory always
	# refit (gw_init only VALIDATED the μ extent), costing 20+ min at
	# fixture scale and ~22 min at MoS2 12×12/P=80 for a byte-identical
	# result.  Override with LORRAX_FORCE_REFIT=1.
	#
	# BISPINOR (2026-08-04).  Reuse used to be gated off for the whole
	# bispinor run — ``_reuse = (not cfg.bispinor) and ...`` — because
	# the bispinor branch also returns ``transverse_wfn_data`` and "that
	# is not on disk".  It does not need to be: that dict is ψ sampled at
	# the transverse centroids (plus meta_T and the centroid table), a
	# pure function of WFN.h5 + the centroid file that the ζ_T fit does
	# not contribute to.  The reuse path REBUILDS it via
	# ``_transverse_wfn_data`` — the same call the fit path makes —
	# instead of returning None.  Returning None here would silently drop
	# Σ^B (rc=0, wrong physics), which is exactly the failure commit
	# 3d89885 fixed on the restart round-trip.  Measured at b600 bispinor
	# (job 7885966) the rebuild costs ~9 s against a 318 s refit.
	#
	# A bispinor reuse is ALL FOUR channels or none: the charge ζ and all
	# three ζ_T must each pass, or every one is refit.  Partial reuse
	# would be defensible (the channels are numerically independent) but
	# it doubles the number of states this cache can be in for no
	# measured gain — the expensive case is the whole set.
	_provenance = _zeta_fit_provenance(
		wfn=wfn, meta=meta, cfg=cfg,
		band_range_left=band_range_left, band_range_right=band_range_right,
		zeta_cutoff=_zeta_cutoff, zeta_vcoul_cutoff=_zeta_vcoul_cutoff,
		write_ibz_only=_write_ibz_only_charge, band_norms=_band_norms,
		vertex_mu_L=0, transverse_identity=_transverse_identity)

	def _provenance_T(mu_L):
		"""The charge stamp with this transverse channel's μ and vertex."""
		return _zeta_fit_provenance(
			wfn=wfn, meta=_meta_T, cfg=cfg,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			zeta_cutoff=_zeta_cutoff, zeta_vcoul_cutoff=_zeta_vcoul_cutoff,
			write_ibz_only=_write_ibz_only_transverse,
			band_norms=_band_norms,
			vertex_mu_L=int(mu_L), transverse_identity=_transverse_identity)

	_zeta_T_paths = {
		mu_L: os.path.join(tmp_dir, f"zeta_q_mu{mu_L}.h5")
		for mu_L in (1, 2, 3)
	}
	_reuse = _zeta_reuse_ok(
		zeta_h5_path, _provenance, centroid_indices, print_fn=print_fn,
		n_rmu_expected=int(meta.n_rmu))
	if _reuse and cfg.bispinor:
		for mu_L, _zeta_T_path in _zeta_T_paths.items():
			if not _zeta_reuse_ok(
					_zeta_T_path, _provenance_T(mu_L), _cent_T_idx,
					print_fn=print_fn,
					n_rmu_expected=int(_meta_T.n_rmu)):
				print_fn(
					f"    [zeta reuse] the transverse ζ for μ_L={mu_L} "
					f"({_zeta_T_path}) is NOT reusable, so the charge ζ is "
					f"not reused either — refitting all four channels.")
				_reuse = False
				break
	if _reuse:
		print_fn("")
		print_fn("  " + "=" * 68)
		print_fn(f"  REUSING the existing ζ at {zeta_h5_path} — FIT SKIPPED.")
		if cfg.bispinor:
			print_fn( "  ...and the three transverse ζ at "
			          "zeta_q_mu{1,2,3}.h5.")
		print_fn( "  isdf_header/zeta_is_done is True, the centroid table")
		print_fn( "  matches, and fit_provenance is identical to this run's")
		print_fn( "  inputs (band windows, cutoffs, solver knobs, source WFN).")
		print_fn( "  Set LORRAX_FORCE_REFIT=1 to refit unconditionally.")
		print_fn("  " + "=" * 68)
		print_fn("")
		if not cfg.bispinor:
			return zeta_h5_path, mem_est, None
		# ψ at r_{μ_T} is NOT on disk here (the restart tensors file is a
		# different mechanism and may not exist yet), so re-sample it.
		# Same call as the fit path — see ``_transverse_wfn_data``.
		print_fn(f"  [bispinor] re-sampling ψ at the {int(_meta_T.n_rmu)} "
		         f"transverse centroids for σ^B (the ζ_T fit is skipped, "
		         f"but Σ^B still needs ψ(r_{{μ_T}})).")
		return zeta_h5_path, mem_est, _transverse_wfn_data(
			wfn, sym, _meta_T, _cent_T_idx, cfg, mesh_xy,
			band_slices, _chunks_T)

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
			distributed_cholesky=cfg.backend.distributed_cholesky,
			distributed_lu=cfg.backend.distributed_lu,
			distrib_la_batched_route=cfg.backend.distrib_la_batched_route,
			zeta_ridge=cfg.backend.zeta_ridge,
			charge_zeta_solve=cfg.backend.charge_zeta_solve,
			distributed_zeta_solve=cfg.backend.distributed_zeta_solve,
			zeta_rcond=cfg.backend.zeta_rcond,
			gflat_chunk_size=int(chunks.get('gflat_chunk_size', 0)),
			write_ibz_only=_write_ibz_only_charge,
			zeta_cutoff_ry=_zeta_cutoff,
			low_mem_bands=bool(cfg.memory.low_mem_bands),
			psi_nmu_fresh=psi_nmu_fresh, psi_mun_fresh=psi_mun_fresh,
		)

	# ── THE ζ RANK CUT'S CLOSURE VERDICT, refused HERE if it must be ──────
	# The four ζ truncation sites live inside jitted kernels, which cannot
	# raise; under ``LORRAX_SPECTRAL_CLOSURE=strict`` they record the finding
	# through a host callback and this is the seam that refuses on it — the
	# same division of labour ``centroid/pivoted_cholesky`` documents for the
	# select kernel.  Placed immediately after the fit and BEFORE ζ is
	# consumed by anything, so ``strict`` stops the run rather than letting W
	# be built on a basis whose span is not point-group invariant.
	#
	# Under the default (``snap``) this prints any firing and continues; the
	# cut has already been moved inside the kernel, DROPPING the straddled
	# block (owner ruling 2026-08-10).  On a deck whose cut falls in a gap it
	# does nothing at all, which is what the Si 6×6×6 ``armF`` (nband=68) arm
	# is expected to show — its tightest cut has a relative gap of 0.315
	# against a tolerance of 1e-6, five decades clear of firing.  That deck is
	# the control for the direction flip: a cut in a gap is unmoved by EITHER
	# direction, so its retained ranks must be bit-unchanged across it.
	_sc_mode = spectral_closure.resolve_mode(
		os.environ.get(spectral_closure.MODE_ENV))
	# Read the findings BEFORE the refusal clears them, so the "clean" line
	# below cannot contradict a snap that just fired one line above it.
	_sc_fired = bool(spectral_closure.pending())
	spectral_closure.raise_if_pending(
		"the ζ fit's rank truncation", mode=_sc_mode, log=print_fn)
	if _sc_mode == "off":
		print_fn("    ζ rank-cut closure: guard is OFF "
		         "(LORRAX_SPECTRAL_CLOSURE=off) — NOT CHECKED, which is an "
		         "absence and not a pass.")
	elif _sc_fired:
		print_fn(f"    ζ rank-cut closure: guard fired above and the "
		         f"straddled block was DROPPED whole (mode={_sc_mode}, "
		         f"direction={spectral_closure.DEFAULT_DIRECTION}).  ζ's "
		         f"retained span is point-group invariant; the rank it "
		         f"carries is LOWER than the one zeta_rcond alone would have "
		         f"chosen, and its κ_eff is correspondingly better.")
	else:
		print_fn(f"    ζ rank-cut closure: ARMED (mode={_sc_mode}, rtol="
		         f"{spectral_closure.DEFAULT_RTOL:.1e}) and SILENT — no cut "
		         f"fell inside a degenerate block of C_q on any q.")

	# ── THE ζ RANK CUT'S CERTIFICATION VERDICT, same seam, same reason ────
	# ``spectral_closure`` asks WHERE the cut landed; this asks whether the
	# cut was allowed to happen at this conditioning at all.  Until
	# 2026-08-22 the ζ truncation printed n_keep/q and kappa/q and gated on
	# NEITHER: a 1776-centroid Si 4×4×4 fit dropped 300+ modes per q at
	# kappa 9.7e9 and delivered Σ_c MAE 54.4 eV at exit 0 with no banner.
	# The gate fires inside the jitted kernels, which cannot raise, so it
	# records through a host callback and refuses HERE — before ζ is
	# consumed by W.  docs/dev/rank_truncation_policy.md owns the policy.
	_rp_mode = rank_criterion.resolve_policy_mode(
		os.environ.get(rank_criterion.POLICY_MODE_ENV))
	_rp_fired = bool(rank_criterion.pending())
	rank_criterion.raise_if_pending(
		"the ζ fit's rank truncation", mode=_rp_mode, log=print_fn)
	if _rp_mode == "off":
		print_fn("    ζ rank-cut certification: gate is OFF "
		         f"({rank_criterion.POLICY_MODE_ENV}=off) — NOT CHECKED, "
		         f"which is an absence and not a pass.")
	elif _rp_fired:
		print_fn(f"    ζ rank-cut certification: FIRED above "
		         f"(mode={_rp_mode}).  The cut bound outside the regime any "
		         f"measurement certifies; the numbers are in the "
		         f"[rank-policy] lines.")
	else:
		print_fn(f"    ζ rank-cut certification: ARMED (mode={_rp_mode}, "
		         f"certified κ ceiling "
		         f"{rank_criterion.KAPPA_CERTIFIED_GRAM:.1e} for the charge "
		         f"Gram, discarded-weight ceiling "
		         f"{rank_criterion.DISCARDED_WEIGHT_MAX:.1e}) and SILENT — "
		         f"either the cut bound nothing, or it bound inside the "
		         f"certified regime.  The transverse channel is "
		         f"UNCERTIFIED and can only raise the weight finding.")

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
		print_fn(f"\n  [bispinor] fitting ζ^{{μ_L=1,2,3}} on current-density "
		         f"centroids: {cfg.paths.centroids_file_current}")
		# ``meta_T``, the centroid table and its μ padding were built ONCE
		# in the bispinor pre-flight (they are inputs to the ζ-reuse
		# decision, which runs before the charge fit).
		meta_curr = _meta_T
		cents_curr_idx = _cent_T_idx
		transverse_wfn_data = _transverse_wfn_data(
			wfn, sym, meta_curr, cents_curr_idx, cfg, mesh_xy,
			band_slices, _chunks_T)
		psi_curr_rmu_Y = transverse_wfn_data['psi_rmu_Y']
		psi_curr_rmuT_X = transverse_wfn_data['psi_rmuT_X']

		# low_mem_bands = true: _transverse_wfn_data already converted
		# psi_curr_rmu_Y/psi_curr_rmuT_X to the two-face carrier internally
		# (SAME PSI_MUN_SPEC/PSI_NMU_SPEC build path the charge channel
		# uses, not a fork) and set them to None -- ONE call site owns the
		# conversion so this branch and the ζ-reuse early return
		# (gw_init.fit_zeta's own "REUSING the existing ζ" path, which
		# also calls _transverse_wfn_data) get an identically-built face
		# carrier rather than each converting it their own way.  Just
		# read the two fields back out here.
		psi_mun_fresh_T = transverse_wfn_data['psi_mun_fresh']
		psi_nmu_fresh_T = transverse_wfn_data['psi_nmu_fresh']
		if cfg.memory.low_mem_bands:
			print_fn("  [bispinor] ψ_T face conversion (low_mem_bands): "
			         "psi_nmu_T/psi_mun_T built from the transverse "
			         "centroid load; both single-axis copies dropped "
			         "before the ζ_T fit.")

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
			zeta_mu_path = _zeta_T_paths[mu_L]
			print_fn(f"  [bispinor] μ_L={mu_L} → {zeta_mu_path}")
			with timing.section(f"gw_jax.zeta_fit_chunked_mu{mu_L}"), \
			     jax_profile.trace_section(f"zeta_fit_mu{mu_L}"):
				fit_zeta_to_h5(
					wfn=wfn, sym=sym, meta=meta_curr,
					centroid_indices=cents_curr_idx,
					mesh_xy=mesh_xy,
					chunk_r=_chunks_T['chunk_r'], output_file=zeta_mu_path,
					psi_rmu_Y=psi_curr_rmu_Y, psi_rmuT_X=psi_curr_rmuT_X,
					low_mem_bands=bool(cfg.memory.low_mem_bands),
					psi_mun_fresh=psi_mun_fresh_T,
					psi_nmu_fresh=psi_nmu_fresh_T,
					band_chunk_size=_chunks_T['band_chunk'],
					q_chunk_size=_chunks_T['q_chunk'],
					bispinor=cfg.bispinor,
					band_range_left=band_range_left,
					band_range_right=band_range_right,
					band_norms=_band_norms,
					distributed_cholesky=cfg.backend.distributed_cholesky,
					distributed_lu=cfg.backend.distributed_lu,
					distrib_la_batched_route=cfg.backend.distrib_la_batched_route,
					zeta_ridge=cfg.backend.zeta_ridge,
					distributed_zeta_solve=cfg.backend.distributed_zeta_solve,
					transverse_zeta_solve=cfg.backend.transverse_zeta_solve,
					transverse_zeta_rcond=cfg.backend.transverse_zeta_rcond,
					gflat_chunk_size=int(_chunks_T.get('gflat_chunk_size', 0)),
					vertex_mu_L=mu_L,
					# Transverse ζ IBZ-write activates whenever the
					# bispinor V_q orchestrator iterates IBZ q's — same
					# gate the charge ζ uses (LORRAX_FORCE_FULL_BZ off),
					# resolved once in the pre-flight so the provenance
					# stamp and this call cannot disagree.
					# Orbit-closure of the transverse centroid set is
					# checked downstream in ``fit_zeta_to_h5``; failure
					# is loud per the bispinor IBZ requirement.
					write_ibz_only=_write_ibz_only_transverse,
					zeta_cutoff_ry=_zeta_cutoff,
				)
			# Stamp this ζ_T so a later run can reuse it — same ordering
			# and same truncating-knob veto as the charge stamp above
			# (fit → mark_zeta_done → stamp; a run killed between the
			# last two leaves a complete-but-unstamped file, which
			# ``_zeta_reuse_ok`` refits).  The μ_L loop was previously
			# unstamped altogether, which is why bispinor ζ from before
			# 2026-08-04 is never reusable.
			if not _trunc and jax.process_index() == 0:
				try:
					from file_io.isdf_header import stamp_fit_provenance
					stamp_fit_provenance(zeta_mu_path, _provenance_T(mu_L))
				except Exception as exc:
					print_fn(f"    [zeta provenance] μ_L={mu_L} not stamped "
					         f"({exc}); this ζ_T will be refit on the next "
					         f"run.")
			barrier(f"zeta_provenance_mu{mu_L}")

	return zeta_h5_path, mem_est, transverse_wfn_data


def _build_head_channel(zeta_io, *, cfg, meta, wfn, bvec, mesh_xy, sym,
                        centroid_indices, vcoul_cutoff_ry, print_fn=print):
	"""Build a q != 0 Coulomb head channel only when a consumer needs it.

	The exact/default combination returns ``None`` before anything is read,
	so the shipped path costs two string comparisons.  A non-default
	``mc_average_placement`` or ``bgw_metal_q0_treatment=bgw_q0shift`` reads
	the ζ head-slot columns (``v_q_g_flat.compute_head_channel_zeta``), pairs
	them with the head-slot ``v`` table, expands the per-IBZ-q scalars onto
	the full BZ, and optionally re-sources the mini-BZ enhancement from a
	BerkeleyGW ``vcoul`` dump.

	``schur_avg`` is refused HERE rather than at the solve, so a deck that
	asks for it fails in the first minute of the run instead of after χ₀.
	"""
	from .head_channel import (PLACEMENT_OFF, HeadChannel,
	                           head_ratio_from_bgw_dump,
	                           refuse_if_unimplemented)

	mode = str(getattr(cfg.head, 'mc_average_placement', PLACEMENT_OFF))
	needs_bgw_q0 = bool(getattr(cfg.head, 'uses_bgw_metal_q0shift', False))
	if mode == PLACEMENT_OFF and not needs_bgw_q0:
		return None
	refuse_if_unimplemented(mode)
	if int(meta.sys_dim) != 3:
		raise NotImplementedError(
			f"mc_average_placement = {mode!r} is a 3D-bulk knob (it moves the "
			f"mini-BZ average of 8*pi/|q+G|^2 onto W's head channel); this "
			f"deck has sys_dim = {int(meta.sys_dim)}.  The 2D f2d->0 "
			f"regularisation already removes the divergence at G=0, so there "
			f"is no q != 0 head slot for the rescale to act on.")

	from .v_q_g_flat import compute_head_channel_zeta
	g_head, table, full_to_irr_idx = compute_head_channel_zeta(
		zeta_io,
		kgrid=meta.kgrid, fft_grid=meta.fft_grid,
		bvec=bvec, cell_volume=meta.cell_volume,
		mesh_xy=mesh_xy, sys_dim=meta.sys_dim,
		bdot=(np.asarray(wfn.bdot, dtype=np.float64)
		      if meta.sys_dim == 0 else None),
		bare_coulomb_cutoff_ry=vcoul_cutoff_ry,
		mc_average_vcoul_body=cfg.head.mc_average_vcoul_body,
		sym=sym, centroid_indices=centroid_indices,
		verbose=(jax.process_index() == 0),
	)
	# IBZ -> full BZ for the per-q scalars.  |q+G| is a class function of
	# the shell and the tied-set MEAN makes <v> one too, so the expansion is
	# a gather, not an unfold: v(q) == v(S q) exactly.
	idx = np.asarray(full_to_irr_idx, dtype=np.int64)
	v_bare = np.asarray(table.v_bare)[idx]
	v_avg = np.asarray(table.v_avg)[idx]
	len2 = np.asarray(table.len2)[idx]
	mult = np.asarray(table.mult)[idx]
	# What V ACTUALLY carries in those slots — the same predicate
	# ``v_q_g_flat`` gates ``v_head_fn`` on.
	v_in_V = (v_avg if (cfg.head.mc_average_vcoul_body and meta.sys_dim == 3)
	          else v_bare)
	hc = HeadChannel(g_head=g_head, v_bare=v_bare, v_avg=v_avg,
	                 v_in_V=v_in_V, mult=mult, len2=len2, mode=mode)
	dump = getattr(cfg.head, 'mc_average_placement_vcoul', None)
	if dump:
		hc = head_ratio_from_bgw_dump(dump, hc, bvec=bvec)
	print_fn(hc.summary())
	return hc


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None, sym=None, centroid_indices=None):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0, head_channel) where V_qmunu has shape (nq, μ, μ)
	(flat-q) and G0 is (n_rmu,) ζ_μ(G=0) at q=0.  ``head_channel`` is a
	``gw.head_channel.HeadChannel`` when the deck sets
	``mc_average_placement`` to something other than ``off``, and ``None``
	otherwise — nothing is computed for it on the default path.  Downstream consumers that need
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

	# The Cartesian reciprocal ROWS come off the vcoul door's geometry, not
	# from a hand-written product.  ``docs/services/vcoul.md`` says it in as
	# many words — "Do not multiply ``wfn.blat * wfn.bvec`` at a call site" —
	# and the reason is the one ``CoulombGeometry``'s own docstring gives: a
	# product every caller has to remember to take is a footgun, because the
	# day one of them passes ``wfn.bvec`` alone every number downstream is
	# off by the lattice constant with no shape error to say so.
	# ``from_wfn`` is duck-typed on ``blat``/``bvec``/``cell_volume``, all
	# three of which ``WfnLoader`` binds off the mf_header.
	#
	# ONLY ``.bvec`` IS TAKEN.  ``meta.cell_volume`` below stays where it is:
	# Ω sets the 1/Ω factor on every v(q+G), so swapping its source is a
	# physics edit, not a plumbing one.  It happens that the two agree
	# exactly — ``Meta.from_system`` and ``CoulombGeometry.from_wfn`` both
	# read ``float(wfn.cell_volume)`` off this same loader, measured
	# bit-identical — but "they agree" is the licence for a later swap, not
	# a reason to make it in a commit about ``bvec``.
	bvec = CoulombGeometry.from_wfn(wfn).bvec

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
	# omission of ``v_head_fn`` costs nothing.  In 3D with
	# ``mc_average_vcoul_body`` enabled the two DO diverge, in the G=0 slot
	# of every q≠0; that path is reachable, so a 3D bispinor deck must not
	# assume the CC tile and the scalar V_q agree.  See the v_q_bispinor
	# CC builder.  We
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
		# REFUSE, do not demote.  This used to print the sentence below and
		# carry on with a scalar V_q — i.e. silently return NON-BISPINOR
		# physics from a deck that asked for bispinor, with Σ^B absent and
		# no symptom in any output.  decisions.md 2026-08-01 rules that a
		# missing capability is a refusal naming what is missing, never a
		# demotion to a different compute path; this is the same class as
		# the FFI entry and the same class as the restart regression
		# 3d89885 fixed.
		_missing = [p for p in zeta_T_paths if not os.path.exists(p)]
		raise FileNotFoundError(
			f"bispinor = true, but {len(_missing)} of 3 transverse ζ files "
			f"are missing:\n  "
			+ "\n  ".join(_missing)
			+ f"\nRefusing to fall back to a scalar V_q: that would drop "
			f"Σ^B and return non-bispinor physics from a bispinor deck "
			f"with no symptom in any output.  Either the ζ fit did not run "
			f"for the transverse channel (check that fit_zeta received "
			f"cfg.centroids_file_current) or the files were removed after "
			f"it did.  Re-run the fit, or set bispinor = false if a scalar "
			f"run is what you want."
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
		# Orbit-closure of the C/T centroid sets is resolved inside
		# ``_resolve_ibz_q_list`` (called per tile by the V_q
		# orchestrator), which falls back to full-BZ on failure and
		# ANNOUNCES it once per centroid set — the charge and the
		# transverse set are separate facts and get separate lines.
		# It used to fall back SILENTLY; see gw/qgrid_symmetry.py.
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
		# HEADER-ONLY (``mesh=None``): no SlabIO handle, no phdf5 FFI, no
		# collective — the same one serial-h5py open the raw
		# ``f['zeta_q_G'].shape[1]`` here did, through the reader that owns
		# the layout.  ``n_rmu_disk`` IS axis 1 in G-flat and axis 2 in
		# r-space, which is the dispatch this line used to assume.  The
		# loader's open-time refusals (zeta_is_done, header-vs-dataset μ,
		# header-vs-dataset ngkmax) come along, and they are a strict
		# SUBSET of what the ``ZetaLoader(zeta_T_paths[0], mesh=mesh_xy)``
		# fifteen lines below already applies to this very file.
		with ZetaLoader(zeta_T_paths[0]) as _z_T0:
			n_rmu_T = int(_z_T0.n_rmu_disk)
		n_rmu_C = int(meta.n_rmu)

		with timing.section("gw_jax.V_q_compute"), \
		     jax_profile.trace_section("V_q_compute_bispinor"):
			# G-flat path: per-q + G-chunked, one orchestrator per
			# four ζ files.  No legacy compute_V_q_tile chooser /
			# μ × ν tiling / in-V_q FFT — see
			# gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5.
			from .v_q_bispinor import compute_V_q_bispinor_g_flat_to_h5
			with ZetaLoader(zeta_h5_path, mesh=mesh_xy,
			                ) as zc, \
			     ZetaLoader(zeta_T_paths[0], mesh=mesh_xy,
			                ) as zt1, \
			     ZetaLoader(zeta_T_paths[1], mesh=mesh_xy,
			                ) as zt2, \
			     ZetaLoader(zeta_T_paths[2], mesh=mesh_xy,
			                ) as zt3:
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
						print_fn=print_fn,
						sym=sym if _use_ibz_bispinor else None,
						centroid_C_idx=_cent_C_idx_for_orchestrator,
						centroid_T_idx=_cent_T_idx_for_orchestrator,
						use_ibz=_use_ibz_bispinor,
						tt_head_correction=bool(
							cfg.head.bispinor_tt_head_correction),
					)

		# Read CC tile + g0 back for downstream restart-state writer.
		# The TT tiles stay on disk; Σ_X^B / Σ_H^B will consume them
		# via BispinorVqReader once those paths land.
		with BispinorVqReader(bispinor_h5_path, mesh_xy,
		                      ) as reader:
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
		# ``mc_average_placement`` is refused on the bispinor builder, not
		# silently skipped.  ``v_q_bispinor`` does not pass ``v_head_fn`` at
		# all, so with ``mc_average_vcoul_body`` on in 3D the bispinor CC tile
		# and the scalar V_q ALREADY diverge in the G=0 slot of every q != 0
		# (see the note at the head of this file).  Adding a second, quieter
		# copy of that divergence is exactly what COULOMB_AVG_ARCHITECTURE.md
		# section 4.6(b) says not to do.
		head_channel = None
		if str(getattr(cfg.head, 'mc_average_placement', 'off')) != 'off':
			raise NotImplementedError(
				"mc_average_placement is not implemented on the bispinor V_q "
				"builder: v_q_bispinor.py passes no v_head_fn, so its CC tile "
				"already carries a different G=0 slot from the scalar V_q at "
				"every q != 0.  Deciding the placement for one builder and not "
				"the other would make that divergence permanent.  Run with "
				"bispinor = false, or land the bispinor v_head_fn first.")
		if bool(getattr(cfg.head, 'uses_bgw_metal_q0shift', False)):
			# The BGW q0 mode does not rescale the bispinor V tile.  It needs
			# only the charge-charge head vector at one finite q, which is
			# carried by the ordinary charge ζ file even in a bispinor run.
			with ZetaLoader(zeta_h5_path, mesh=mesh_xy) as zeta_io:
				with mesh_xy:
					head_channel = _build_head_channel(
						zeta_io, cfg=cfg, meta=meta, wfn=wfn, bvec=bvec,
						mesh_xy=mesh_xy, sym=sym,
						centroid_indices=_cent_C_idx_for_orchestrator,
						vcoul_cutoff_ry=vcoul_cutoff_ry,
						print_fn=print_fn)
	else:
		# Scalar (non-bispinor) path.  ``compute_all_V_q`` dispatches on
		# the on-disk ζ layout: G-flat (the only thing fit_zeta writes)
		# routes to ``v_q_g_flat.compute_all_V_q_g_flat``; any other
		# layout raises.  ``ZetaLoader`` is the V_q reader of record —
		# it serves the writer's per-q WFN.h5-style G-sphere directly.
		with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
			with ZetaLoader(zeta_h5_path, mesh=mesh_xy,
			                ) as zeta_io:
				_cent_idx_np = (
					np.asarray(jax.device_get(centroid_indices),
					           dtype=np.int32)
					if centroid_indices is not None else None)
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
						centroid_indices=_cent_idx_np,
						g_chunk_size=int(cfg.memory.vq_g_chunk_size),
					)
					# The q != 0 head channel, for ``mc_average_placement``.
					# Gated on the mode so the default path neither reads ζ a
					# second time nor compiles a single extra kernel.  It sits
					# INSIDE the loader scope because that is the only place
					# ζ is open, and before ``V_q_raw`` is padded so the μ
					# extents agree by construction.
					head_channel = _build_head_channel(
						zeta_io, cfg=cfg, meta=meta, wfn=wfn, bvec=bvec,
						mesh_xy=mesh_xy, sym=sym,
						centroid_indices=_cent_idx_np,
						vcoul_cutoff_ry=vcoul_cutoff_ry,
						print_fn=print_fn)

	# Write G0 = ζ_μ(G=0) at q=0 back to zeta file via SlabIO's deferred
	# attr path (small; rank-0-only after MPI-IO file is closed).
	# ``common.collectives.gather_to_host`` is the sanctioned L3 gather and
	# is what ``_slab_io_allgather._to_host`` was a private copy of.  This
	# import used to reach straight into the allgather backend, bypassing
	# every one of the seven refusals that guarded that tier -- an eighth,
	# ungated door.  G0 is (nq, mu), mu-class not mu^2-class, so the gather
	# itself is not the doctrine violation; the unguarded private import
	# was.  Same dispatch, public name.
	from common.collectives import gather_to_host as _gather_to_host
	G0_gathered = _gather_to_host(G0_all)
	# The door's documented sequence (``write_g0_mu``: close -> barrier ->
	# rank-0 write -> barrier) is now LITERALLY what this caller does.
	# ``gather_to_host`` above is itself collective and synchronized this
	# point in practice, but the step-3 blind audit (Arm B §7) found the
	# pre-write barrier existing only by that side effect — one explicit
	# line is cheaper than a sequence that is true by coincidence.
	barrier("g0_pre_write")
	if jax.process_index() == 0:
		# Clip the μ axis to the LOGICAL extent: files on disk store
		# logical extents so they re-read identically on any process
		# count (G0_all is at the in-memory padded extent, pad
		# entries exact zeros).  THE CLIP STAYS HERE, at the call site,
		# because only this caller knows which of its axes is μ; the door
		# takes ``n_rmu_expected`` and turns "the caller clipped
		# correctly" from a convention into a check.
		_g0_np = np.asarray(G0_gathered)[..., :int(meta.n_rmu)]
		# THE ONE sanctioned post-close serial append into a ζ file.  The
		# rank-0 gate and the barrier below are the CALLER's on purpose
		# (``write_g0_mu``'s docstring: putting the gate inside would make
		# it look safe to call anywhere, and the failure that hides is
		# silent concurrent-writer corruption).  Ordering is unchanged:
		# after ``gather_to_host``, after every collective handle has
		# closed, rank-0 only, barrier'd.
		write_g0_mu(zeta_h5_path, _g0_np, n_rmu_expected=int(meta.n_rmu))
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
	# Per-q hermiticity above is a NEIGHBOURING property, not the one the
	# BSE kernel rests on.  That one is q↔−q conjugate reciprocity,
	# V_q = conj(V_{−q}) (equivalently: ifft_q(V_q) is REAL), and it is
	# independent of hermiticity in both directions.  V_q passes the
	# hermiticity gate at 3.0e-16 on every fixture measured while failing
	# the reciprocity at 5.7e-3 (armA_base480, 2026-08-07) -- and a q=0
	# check could not have seen it either way, because -0 == 0 makes the
	# condition collapse to "V[0] is real", which holds at 3.7e-16.
	# V is the BARE Coulomb: static and analytic, no frequency dependence
	# anywhere, so the dynamical/Kramers-Kronig caveat that applies to a
	# real-axis W does not apply here at all.  Reciprocity is simply true.
	# TOLERANCE from the MEASURED floor, not from eps: these tiles span
	# |A| in [2.6, 4.7e6] and the residual is set by cancellation among
	# large intermediates, not by eps*max|A| (= 1.0e-9 here).  The
	# empirical floor is the orbit-closed IBZ arm: MEASURED 1.16e-7
	# (armB_orbit504, 2026-08-07), with the per-element relative residual
	# falling as |A| rises, which is the round-off signature.  The DIRECT
	# arm instead sits at 1.5e-3 per-element relative and FLAT in |A| --
	# systematic, not round-off.  1e-5 is ~90x above the floor and ~400x
	# below that break.
	#
	# WHAT THAT FLOOR IS *NOT*.  This comment used to say that on the
	# orbit-closed IBZ arm "the unfold builds V_{-q} from V_q by symmetry
	# so reciprocity holds BY CONSTRUCTION", and read the 1.16e-7 as an
	# arithmetic floor.  Both halves are false.  The unfold applies a
	# SPATIAL operation; reciprocity is a statement about complex
	# conjugation.  They coincide only if the finite ISDF zeta basis is
	# point-group covariant -- an unstated assumption, and MEASURED FALSE
	# by 1.240e-02 at Gamma on the Na 8x8x8 SOC c464 deck.  So 1.16e-7 is
	# a measurement of zeta covariance on ONE deck, not a floor any deck
	# inherits.
	#
	# AND THIS GATE IS BLIND WHERE THAT DEFECT IS LARGEST.  At a q with
	# q == -q (Gamma, and every TRIM of an even mesh) the condition
	# collapses to "V_q is real", which the analytic assembly satisfies at
	# machine epsilon whatever the covariance does: 3.9e-17 at Gamma and
	# 6.4e-17 at H on the deck whose covariance residual there is 1.2e-02
	# and 2.4e-02.  The discriminating statistic is the little-group
	# covariance of the IBZ PARENTS, measured at the unfold sites in
	# ``v_q_g_flat``/``screening``/``screening_bse`` through
	# ``QgridTrsPolicy.measure_covariance`` and reported by
	# ``sanity.report_parent_covariance``.  Do NOT tighten the rtol here
	# to compensate; this statistic is measuring a projection.
	sanity.check_q_conjugate_reciprocity(
		"V_q[all q]", V_q_raw, tuple(meta.kgrid), rtol=1e-5,
		print_fn=print_fn)
	sanity.check_finite("V_q G0 (ζ_μ(G=0) at q=0)", G0, print_fn=print_fn)
	return V_qmunu, G0, head_channel


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
	mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None,
):
	"""ISDF pipeline (non-restart path reads top-to-bottom):

	  1. ``plan_gflat_chunks`` → band/r/q/G-flat chunk plan.
	  2. ``load_centroids_band_chunked`` → ψ at centroids for [b0, b4).
	  3. ``fit_zeta`` → ζ.h5 (consumes ψ slices for pair density).
	  4. ``compute_V_q`` → V_qmunu, G0 (reads ζ from disk).
	  5. Flush V_q / G0 / enk + W0 placeholder to restart H5 (mode="w").
	  6. ``build_wavefunctions`` → 4-copy Wavefunctions bundle (reuses ψ).
	  7. Append ``psi_full_y`` (= wfns.psi_yr) to restart H5 (mode="a").

	Returns SimpleNamespace(V_qmunu, wf_bundle).
	"""
	from file_io import write_restart_state_to_h5
	from common.wfn_transforms import load_centroids_band_chunked
	# The deck's write/skip decision for tmp/isdf_tensors_*.h5, taken and
	# announced in ONE place (``write_restart_tensors``).
	from .gw_output import restart_tensor_writes_enabled
	# The q-storage decision + the hand-off slot the V_q producer deposits
	# its pre-unfold block into.  Imported here rather than at module scope
	# for the same reason the writer is: this function is the only caller.
	from .restart_q_storage import (resolve_restart_q_storage_for_run,
	                                take_pre_unfold)

	# THE DRIVER-ENTRY MIRROR of the parse-time low_mem_bands envelope
	# check (``LorraxConfig.from_input_file`` already ran this on the
	# production path — ``gw_jax.main`` never reaches this function without
	# it).  No-op there, so this costs nothing on the path it duplicates;
	# what it buys is a hand-built ``cfg`` (a test harness, a future direct
	# caller) refusing HERE, before the chunk planner / ISDF fit / restart
	# tensor read below, rather than reaching a bare AttributeError deep in
	# an unported consumer.  Same two-call-site shape as
	# ``refuse_unimplemented_compute_mode`` (parser + ``compute_sigma_xc``)
	# for the same reason.  Replaces the two bispinor-only ad hoc guards
	# this function used to carry on its own fresh/restart branches: this
	# single call covers all four table rows, earlier, on both branches.
	refuse_unsupported_low_mem_bands(cfg)
	# Same shape, same two-call-site reason: parser-altitude coverage is
	# duplicated here for a hand-built cfg.  No-op at the default (false).
	refuse_unsupported_bispinor_tt_head_correction(cfg)

	if not cfg.restart:
		from common.wfn_transforms import get_enk_bandrange

		with mesh_xy:
			# Plan chunk sizes ONCE for the CHARGE channel — the single
			# production planner owns band_chunk / chunk_r / q_chunk /
			# gflat_chunk_size, the rank floor P_min, and the binding-stage
			# report.  ``_plan_gflat_chunks_for_channel`` is the one call
			# site both this (charge) and ``fit_zeta``'s own transverse
			# re-plan (μ_T ≈ μ_C/3) go through, so the two channels cannot
			# drift apart.
			chunks, gflat_plan = _plan_gflat_chunks_for_channel(
				meta=meta, cfg=cfg, band_slices=band_slices, mesh_xy=mesh_xy,
				is_bispinor=bool(cfg.bispinor), print_fn=print0)

			# Load centroid ψ once for the full [b0, b4) range; reused by
			# both the zeta fit (sliced into halves internally) and the
			# downstream Wavefunctions bundle.
			with timing.section("gw_jax.load_centroid_wfns"):
				psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
					wfn, sym, meta, centroid_indices, cfg.bispinor, mesh_xy,
					band_range=band_slices.full_range,
					band_chunk_size=chunks['band_chunk'],
				)

			# ``low_mem_bands = true``: convert to the two-face carrier
			# RIGHT AWAY and drop BOTH single-axis ψ copies before the ζ
			# fit even starts.  Neither has a consumer left in the fit:
			# the CCT Gram build (STEP 2 of fit_zeta_to_h5) reads
			# psi_nmu_fresh/psi_mun_fresh via a distributed SUMMA GEMM
			# (docs/architecture/zeta_fit_face_psi_cct.md), and the
			# r-chunk loop's band contraction (STEP 6) now ALSO reads
			# psi_mun_fresh directly, one band-chunk at a time
			# (isdf.core._z_q_face — the r-chunk section of the same
			# design note), instead of a resident single-axis X-form
			# slice.  Both faces are FREE resharding constraints (no
			# transpose collective, no gather — see
			# wavefunction_bundle.build_wavefunctions_face's docstring,
			# whose derivation this mirrors exactly, just earlier and
			# reused across the fit AND the post-fit bundle instead of
			# rebuilt for each).  bispinor + low_mem_bands is already
			# refused above (refuse_unsupported_low_mem_bands), so this
			# never collides with the bispinor loader branch below.
			psi_nmu_fresh = None
			psi_mun_fresh = None
			if cfg.memory.low_mem_bands:
				from jax.sharding import NamedSharding
				from .wavefunction_bundle import PSI_MUN_SPEC, PSI_NMU_SPEC
				with mesh_xy:
					psi_nmu_fresh = jax.lax.with_sharding_constraint(
						psi_rmu_Y, NamedSharding(mesh_xy, PSI_NMU_SPEC))
					psi_mun_fresh = jax.lax.with_sharding_constraint(
						jnp.conj(psi_rmuT_X).transpose(0, 3, 1, 2),
						NamedSharding(mesh_xy, PSI_MUN_SPEC))
				del psi_rmu_Y, psi_rmuT_X
				psi_rmu_Y = None
				psi_rmuT_X = None
				print0("  ψ face conversion (low_mem_bands): psi_nmu/"
				       "psi_mun built from the fresh load; BOTH "
				       "single-axis copies (psi_rmu_Y, psi_rmuT_X) "
				       "dropped before the ζ fit -- the r-chunk loop "
				       "now reads psi_mun_fresh directly, one "
				       "band-chunk at a time (isdf.core._z_q_face).")

			zeta_path, mem_est, transverse_wfn_data = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, band_slices, tmp_dir,
				psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print0,
				psi_nmu_fresh=psi_nmu_fresh, psi_mun_fresh=psi_mun_fresh)
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

			# ``low_mem_bands = true`` (layout="face"): convert the fit's
			# two single-axis ψ copies to the two face layouts NOW, before
			# V begins, and DROP the single-axis arrays.  The face copies
			# (2*S/(Px*Py) total) then stay resident through V/W/Σ
			# unchanged — there is no later "narrow four copies to two"
			# step the way the legacy path has.  This lowers the V/W/Σ
			# baseline; it does NOT lower the ζ-fit's own peak (that needs
			# a lower-memory fit input contract of its own — audit report
			# census row "Fresh centroid load/liveness").
			#
			# ``low_mem_bands = false`` (layout="legacy", the default) is
			# UNTOUCHED below: psi_rmu_Y/psi_rmuT_X stay alive exactly as
			# before, and the four-copy bundle is still built AFTER
			# compute_V_q, at its original call site.
			wfns = None
			wfns_transverse = None
			if cfg.memory.low_mem_bands:
				# psi_nmu_fresh/psi_mun_fresh were already built (and
				# psi_rmu_Y already dropped) BEFORE the ζ fit, above --
				# they are not re-derived here.  wavefunctions_face_from_
				# restart wraps them into the bundle directly (no
				# resharding constraint, no transpose collective: mirrors
				# build_wavefunctions_face's own construction, just with
				# the faces already built).
				from .wavefunction_bundle import wavefunctions_face_from_restart
				from common.wfn_transforms import (
					get_enk_bandrange as _get_enk_bandrange_early)
				_enk_full_face, _ = _get_enk_bandrange_early(
					wfn, sym, band_slices.full_range,
					(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)
				with timing.section("gw_jax.wavefunction_setup"):
					wfns = wavefunctions_face_from_restart(
						psi_nmu_fresh, psi_mun_fresh, enk_full=_enk_full_face,
						slices=band_slices, mesh_xy=mesh_xy)
				print0(f"  Wavefunctions built (b0:b4={band_slices.nb_full} "
				       f"bands, face layout: psi_nmu/psi_mun; "
				       f"low_mem_bands=true) — V_q's baseline is the "
				       f"face floor alone, no fit-input carryover")
				# No deletion needed here any more.  Both single-axis
				# copies (psi_rmu_Y, psi_rmuT_X) were already dropped
				# BEFORE the ζ fit (above): psi_rmuT_X's only remaining
				# consumer, the r-chunk loop's band contraction, now
				# reads psi_mun_fresh directly (isdf.core._z_q_face),
				# so there is nothing left resident from the fit input
				# for V_q's baseline to inherit.

				# Bispinor + low_mem_bands (2026-08-23): build the
				# transverse-centroid Σ^B bundle from the SAME face
				# carrier ``fit_zeta`` already built
				# (``transverse_wfn_data['psi_mun_fresh']``/
				# ``['psi_nmu_fresh']``) -- reused, not rebuilt,
				# mirroring the charge bundle's own reuse above.  The
				# SAME ``enk_full``/``band_slices`` apply (the mean-field
				# bands are a system property, not per-centroid-set --
				# exactly what the legacy branch below does too).
				if transverse_wfn_data is not None:
					with timing.section("gw_jax.wavefunction_setup"):
						wfns_transverse = wavefunctions_face_from_restart(
							transverse_wfn_data['psi_nmu_fresh'],
							transverse_wfn_data['psi_mun_fresh'],
							enk_full=_enk_full_face,
							slices=band_slices, mesh_xy=mesh_xy)
					print0(f"  [bispinor] σ^B-side Wfns built on "
					       f"n_rmu_T={transverse_wfn_data['meta'].n_rmu} "
					       f"transverse centroids (face layout; "
					       f"low_mem_bands=true)")

			# P4 — pre-V_q.  Whatever's still in HBM after fit_zeta
			# returns forms the persistent baseline that V_q's transient
			# peak stacks on top of.  Same env gate as the ζ-fit probes
			# (LORRAX_MEM_DEBUG=1).  Round-1 addition.
			from gw.isdf_fitting import mem_probe as _mem_probe
			_mem_probe("pre_v_q")
			V_qmunu, G0, head_channel = compute_V_q(
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
			#
			# ``write_restart_tensors = false`` skips this and the ψ
			# append and the centroid stamp TOGETHER — one decision,
			# taken once and announced once by
			# ``gw_output.restart_tensor_writes_enabled``.  Writing
			# some of the datasets and not the others would produce a
			# file the band-window and W0_ready guards accept and the
			# BSE loader then trips over one dataset later.
			#
			# THE GUARDED BLOCK CONTAINS COLLECTIVES (the SlabIO writes
			# and ``barrier("restart_centroid_stamp")``), so the
			# predicate MUST be rank-invariant or the skipping ranks
			# hang the writing ones.  It is: a deck key, parsed from a
			# file every rank reads, with no env override and no
			# probe.  ``restart_tensor_writes_enabled``'s announcement
			# is rank-0-only but prints rather than communicating, so
			# it adds no collective of its own.
			_write_restart = restart_tensor_writes_enabled(
				cfg, tensors_filename)
			# THE q-STORAGE DECISION, TAKEN ONCE, FOR BOTH TENSORS.
			# ``resolve_restart_q_storage`` is rank-invariant for the same
			# reason ``restart_tensor_writes_enabled`` is — a deck key and
			# a centroid file every rank reads — so it adds no collective
			# and cannot make ranks disagree about whether to write.
			# The decision is taken even when the writes are suppressed:
			# it costs nothing and the announcement is the only place a
			# log says which q-set the file would have been on.
			_qirr = resolve_restart_q_storage_for_run(
				cfg, sym=sym, centroid_indices=centroid_indices,
				fft_grid=meta.fft_grid, print_fn=print0)
			if _write_restart:
				from file_io import coulomb_policy_from_config
				write_restart_state_to_h5(
					tensors_filename,
					n_rmu_logical=int(meta.n_rmu),
					# THE COULOMB-KERNEL POLICY, stamped with the tensors it
					# describes.  V_qmunu is reused verbatim by every later
					# restart and compute_V_q never re-runs, so without this
					# an averaging-policy change is inherited in silence with
					# every other guard passing.
					coulomb_policy=coulomb_policy_from_config(cfg, meta),
					V_qmunu=V_qmunu, G0_mu_nu=G0, enk_full=enk_full,
					init_W0=True, mesh=mesh_xy,
					mode="w", kgrid=tuple(int(v) for v in meta.kgrid),
					# Stamp the band window + n_rmu so a later restart
					# under a CHANGED window fails loudly instead of
					# silently misindexing Sigma (job 7874375).
					band_slices=band_slices,
					# ONE resolution, bound to the wedge the V_q producer
					# offered.  ``take_pre_unfold`` REMOVES it, so the W0
					# writer downstream cannot be handed V's block.
					qirr=_qirr.with_capture(
						take_pre_unfold("V_qmunu")),
				)

			# ``low_mem_bands = true``: ``wfns`` was already built (face
			# layout) right after fit_zeta, before V_q — see above.  The
			# legacy path is UNTOUCHED: build the four-copy bundle here,
			# at its original call site, exactly as before this key
			# existed.
			if not cfg.memory.low_mem_bands:
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
			#
			# FACE LAYOUT writes the SAME "psi_full_y" dataset (same
			# name, same (nk, nb, ns, mu) shape/axis order the BSE/
			# downfold readers expect — ``wfns.psi_nmu``'s axis order
			# (nk, n, s, mu) already matches it byte for byte) plus one
			# ADDITIVE "psi_full_y_mun" dataset carrying the second face
			# — SlabIO's ``write_slab`` sources bytes from whatever
			# sharding the array already has (no reshard, no gather), so
			# writing from a 2-D-sharded face array costs the writer
			# nothing beyond the extra dataset.  On restart, this trades
			# 2x ψ bytes on disk for ZERO reshard collectives on read
			# (two direct hyperslab reads instead of one read plus an
			# x<->y mesh-axis transpose) — see
			# ``load_restart_state_from_h5``.
			if _write_restart:
				if cfg.memory.low_mem_bands:
					# Face layout writes BOTH channels' face pairs
					# (2026-08-23): the charge pair
					# (psi_full_y/psi_full_y_mun) and, on a bispinor
					# deck, the transverse pair
					# (psi_full_y_transverse/psi_full_y_transverse_mun)
					# — sourced from the transverse face carrier's own
					# psi_nmu/psi_mun (the same axis-order identity the
					# charge pair relies on), clipped at the TRANSVERSE
					# centroid count.  SlabIO writes each face's owned
					# shards directly; no reshard, no gather.
					write_restart_state_to_h5(
						tensors_filename,
						n_rmu_logical=int(meta.n_rmu),
						psi_full_y=wfns.psi_nmu,
						psi_full_y_mun=wfns.psi_mun,
						mesh=mesh_xy, mode="a",
						psi_full_y_transverse=(
							wfns_transverse.psi_nmu
							if wfns_transverse is not None else None),
						psi_full_y_transverse_mun=(
							wfns_transverse.psi_mun
							if wfns_transverse is not None else None),
						n_rmu_transverse_logical=(
							int(transverse_wfn_data['meta'].n_rmu)
							if transverse_wfn_data is not None else None),
					)
				else:
					write_restart_state_to_h5(
						tensors_filename,
						n_rmu_logical=int(meta.n_rmu),
						psi_full_y=wfns.psi_yr, mesh=mesh_xy,
						mode="a",
						psi_full_y_transverse=(
							wfns_transverse.psi_yr
							if wfns_transverse is not None else None),
						n_rmu_transverse_logical=(
							int(transverse_wfn_data['meta'].n_rmu)
							if transverse_wfn_data is not None else None),
					)
				# Stamp the centroid tables' CONTENT hashes so a restart
				# can verify the quadrature points, not just their counts
				# (see ``_centroid_table_md5``; audit fix/zq 2026-07-28).
				# Rank 0 only, after the collective writer released the
				# file; a failed stamp is non-fatal — the restart-side
				# guard then warns about the missing attr instead of
				# verifying.
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
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		# ``mc_average_placement`` changes the finite-q W operator and still
		# cannot reuse a restart V.  ``bgw_metal_q0_treatment``, by contrast,
		# needs only one finite-q head vector from the already fitted ζ file;
		# build that channel below without repeating either ISDF fit or V_q.
		head_channel = None
		if str(getattr(cfg.head, 'mc_average_placement', 'off')) != 'off':
			raise RuntimeError(
				"mc_average_placement = "
				f"{cfg.head.mc_average_placement!r} with restart = true: the "
				"restart path reuses V_qmunu verbatim and never re-runs "
				"compute_V_q, so the head-channel ζ columns the rescale needs "
				"are not available.  Rerun with restart = false (the placement "
				"changes W, so an inherited W0 would be the wrong object "
				"anyway).")
		# bispinor + low_mem_bands (and the other three envelope rows)
		# already refused at this function's entry
		# (refuse_unsupported_low_mem_bands), before either restart branch
		# above ran.
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices,
				n_rmu_logical=int(meta.n_rmu),
				low_mem_bands=bool(cfg.memory.low_mem_bands))
			V_qmunu = rs.V_qmunu
			print0("  Loaded restart tensors from H5.")
			# COULOMB-KERNEL POLICY DISCLOSURE.  Loud on a mismatch, one
			# line on a match, and a NAMED "not stamped" on a legacy file —
			# the three are different facts and the log says which.  A
			# warning rather than a refusal: a policy change makes the
			# stored V a legitimate tensor built under another convention,
			# and which one the operator wants is not this seam's call.
			# What is removed is the silence.
			from file_io import describe_coulomb_policy_match
			print0(describe_coulomb_policy_match(tensors_filename, cfg, meta))
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
			if cfg.memory.low_mem_bands:
				sanity.check_finite("restart ψ (psi_full_y -> psi_nmu)",
				                    rs.psi_nmu, print_fn=print0)
				sanity.check_finite("restart ψ (psi_full_y_mun)",
				                    rs.psi_mun, print_fn=print0)
			else:
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
			if cfg.memory.low_mem_bands:
				# Both faces were already read at their OWN specs
				# (P(None,'x',None,'y') / P(None,None,'x','y')) directly
				# off disk by ``load_restart_state_from_h5`` — no
				# transpose collective, no y-only full-band replica
				# staged in between.
				from .wavefunction_bundle import wavefunctions_face_from_restart
				wfns = wavefunctions_face_from_restart(
					rs.psi_nmu, rs.psi_mun, enk_full=rs.enk_full,
					slices=band_slices, mesh_xy=mesh_xy)
				print0(f"  Wavefunctions loaded from restart (face layout: "
				       f"psi_nmu/psi_mun; low_mem_bands=true)")
			else:
				wfns = build_wavefunction_bundle(
					wfn, sym, meta, band_slices, mesh_xy,
					psi_rmu_Y=rs.psi_rmu_Y, psi_rmuT_X=rs.psi_rmuT_X,
					enk_full=rs.enk_full, print_fn=print0)
			if bool(getattr(cfg.head, 'uses_bgw_metal_q0shift', False)):
				zeta_path = os.path.join(tmp_dir, "zeta_q.h5")
				bvec = CoulombGeometry.from_wfn(wfn).bvec
				vcoul_cutoff_ry = (
					float(wfn.ecutwfc)
					if cfg.head.bare_coulomb_cutoff is None
					else float(cfg.head.bare_coulomb_cutoff))
				cent_idx_np = np.asarray(
					jax.device_get(centroid_indices), dtype=np.int32)
				with ZetaLoader(zeta_path, mesh=mesh_xy) as zeta_io:
					with mesh_xy:
						head_channel = _build_head_channel(
							zeta_io, cfg=cfg, meta=meta, wfn=wfn,
							bvec=bvec, mesh_xy=mesh_xy, sym=sym,
							centroid_indices=cent_idx_np,
							vcoul_cutoff_ry=vcoul_cutoff_ry,
							print_fn=print0)
				print0(
					"  [bgw q0 provenance] restart reused V_q and ζ fit; "
					"loaded only the finite-q0 head channel from tmp/zeta_q.h5.")
			# Bispinor restart: the transverse-centroid ψ round-trips
			# through the per-channel ``psi_full_y_transverse`` dataset
			# (written 2026-07-27+).  Anything missing or mismatched is
			# a LOUD refusal — before this, ``wfns_transverse=None``
			# flowed into the Σ kernels whose Σ^B fold-in is a silent
			# no-op on None (rc=0, wrong physics: Σ^B dropped).
			wfns_transverse = None
			if cfg.bispinor:
				_have_T_psi = (
					getattr(rs, 'psi_nmu_transverse', None) is not None
					if cfg.memory.low_mem_bands
					else getattr(rs, 'psi_rmu_Y_transverse', None)
					is not None)
				if not _have_T_psi:
					raise ValueError(
						f"bispinor restart: {tensors_filename} has no "
						f"transverse ψ for this layout "
						f"({'face pair psi_full_y_transverse[_mun]' if cfg.memory.low_mem_bands else 'psi_full_y_transverse'}).  "
						f"It was written either by a scalar run, by a "
						f"LORRAX predating the bispinor restart "
						f"round-trip (legacy 2026-07-27, face "
						f"2026-08-23), or by the other layout.  Σ^B "
						f"needs ψ at the transverse centroids; rerun "
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
				if cfg.memory.low_mem_bands:
					from .wavefunction_bundle import (
						wavefunctions_face_from_restart)
					sanity.check_finite(
						"restart transverse ψ (psi_full_y_transverse, "
						"face)", rs.psi_nmu_transverse, print_fn=print0)
					wfns_transverse = wavefunctions_face_from_restart(
						rs.psi_nmu_transverse, rs.psi_mun_transverse,
						enk_full=rs.enk_full, slices=band_slices,
						mesh_xy=mesh_xy)
				else:
					sanity.check_finite(
						"restart transverse ψ (psi_full_y_transverse)",
						rs.psi_rmu_Y_transverse, print_fn=print0)
					wfns_transverse = build_wavefunction_bundle(
						wfn, sym, meta, band_slices, mesh_xy,
						psi_rmu_Y=rs.psi_rmu_Y_transverse,
						psi_rmuT_X=rs.psi_rmuT_X_transverse,
						enk_full=rs.enk_full, print_fn=print0)
				print0(f"  [bispinor] σ^B-side Wfns rebuilt from restart "
				       f"(layout="
				       f"{'face' if cfg.memory.low_mem_bands else 'legacy'}, "
				       f"n_rmu_T={int(rs.n_rmu_transverse_disk)} "
				       f"transverse centroids)")

	return SimpleNamespace(
		V_qmunu=V_qmunu,
		wf_bundle=wfns,
		wf_bundle_transverse=wfns_transverse,
		head_channel=head_channel,
	)

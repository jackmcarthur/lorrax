"""ISDF fitting orchestration for LORRAX GW.

  fit_zeta / compute_V_q — pipeline steps
  prepare_isdf_and_wavefunctions — top-level orchestrator called by main()

Chunk sizing (band_chunk / r_chunk / q_chunk / gflat_chunk_size) is owned
entirely by :func:`gw.gflat_memory_model.plan_gflat_chunks` — the single
production planner (persistent floor + max over five stage transients).
"""
import hashlib
import json
import os
import threading
from dataclasses import dataclass, replace
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
from common.four_current_model import (
	RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION,
	resolve_four_current_representation,
)
from runtime import debug_print_enabled
from file_io.wfn_basis import centroid_table_md5 as _centroid_table_md5

# Canonical env grammar for this layer.  ``gw_config`` is deliberately
# jax-free, so importing it here adds nothing to the import graph that
# the declarations below do not already add.  See the module comment in
# gw_config for why this vocabulary is duplicated rather than imported from
# ``isdf.core`` (which imports jax) — and for the drift gate that keeps the
# copies identical.
from .gw_config import (
	env_bool,
	active_zeta_truncating_knobs,
	classify_xla_pool,
	refuse_unsupported_bispinor_tt_head_correction,
	refuse_unsupported_bispinor_gw,
	resolve_xla_gpu_memory_env,
	uses_coupled_photon_head,
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
# TWO NAMES, THREE FORMER RAW-h5py SITES.  ``probe_zeta_file`` replaced
# the two hand-written copies of the dataset-name/μ-axis dispatch below
# (``_check_zeta_h5_matches_basis`` and ``_zeta_reuse_ok``) — that tuple is
# spelled ONCE now, in the service, and this comment deliberately does not
# spell it a fourth time; ``ZetaLoader`` replaced the raw ``n_rmu_T`` shape
# read.  G=0 remains an in-memory view of the canonical ζ tensor; it is not
# persisted as a duplicate dataset.
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from zeta_loader import ZetaLoader, probe_zeta_file              # noqa: E402
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
_ZETA_PROVENANCE_SCHEMA = 2

# A restart/MPA receipt is deliberately not the raw zeta provenance JSON.
# That JSON owns two launch-local locator fields; the canonical WFN content
# fingerprint supersedes both when the fit crosses an artifact boundary.
CHARGE_ZETA_IDENTITY_SCHEME = (
	"charge-zeta-v1:canonical-provenance+bound-wfn")
_ZETA_PROVENANCE_LOCATOR_FIELDS = frozenset(("wfn_file", "wfn_bytes"))


def charge_zeta_identity(provenance_json, *, wfn,
		wfn_fingerprint_binding=None):
	"""Return the path-independent identity of one charge-zeta fit.

	The zeta provenance owner decides which fields are mere locators.  MPA and
	restart I/O receive only this opaque two-string receipt and therefore never
	maintain a second table of zeta semantics.  Every non-locator field remains
	in the digest, while the source WFN is represented by the incumbent
	``common.parallel_transport`` content identity.
	"""
	try:
		semantic = json.loads(str(provenance_json))
	except (TypeError, ValueError) as exc:
		raise ValueError(
			"charge_zeta_identity requires canonical zeta provenance JSON") from exc
	if not isinstance(semantic, dict):
		raise ValueError("charge_zeta_identity provenance must decode to an object")
	if int(semantic.get("schema", -1)) != _ZETA_PROVENANCE_SCHEMA:
		raise ValueError(
			"charge_zeta_identity requires current zeta provenance schema "
			f"{_ZETA_PROVENANCE_SCHEMA}; got {semantic.get('schema')!r}")
	for key in _ZETA_PROVENANCE_LOCATOR_FIELDS:
		semantic.pop(key, None)
	from common.parallel_transport import (
		WFN_FINGERPRINT_SCHEME, fingerprint_from_binding, wfn_fingerprint)
	wfn_digest = (
		wfn_fingerprint(wfn) if wfn_fingerprint_binding is None else
		fingerprint_from_binding(wfn_fingerprint_binding, wfn))
	semantic["wfn_fingerprint_scheme"] = WFN_FINGERPRINT_SCHEME
	semantic["wfn_fingerprint"] = wfn_digest
	canonical = json.dumps(
		semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
	digest = hashlib.sha256()
	digest.update(CHARGE_ZETA_IDENTITY_SCHEME.encode("ascii"))
	digest.update(b"\0")
	digest.update(canonical.encode("utf-8"))
	return {
		"scheme": CHARGE_ZETA_IDENTITY_SCHEME,
		"digest": digest.hexdigest(),
	}


def _zeta_fit_provenance(*, wfn, meta, cfg, band_range_left, band_range_right,
                         logical_band_stop, zeta_cutoff, zeta_vcoul_cutoff,
                         write_ibz_only, band_norms, vertex_mu_L=0,
                         carrier_bispinor=None, carrier_lift=None,
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

	``band_range_*`` are the storage ranges handed to the fit.  Their right
	edge may include exact-zero mesh padding, so schema 2 stamps only the
	corresponding ``*_logical`` ranges, clipped at ``logical_band_stop``.
	That stop is the already-resolved ``zeta_nband`` when narrowed and
	``Meta.b_id_4_user`` otherwise.  A legacy schema-1 stamp has only the
	storage ranges and cannot prove whether its last rows were padding or
	physical bands; :func:`_zeta_reuse_ok` therefore fails it closed.

	``transverse_identity`` pins every raw-four-spinor ζ set to its transverse
	centroid table and solve gauge.  The explicit Pauli-reference charge fit is
	the one exception: that vertex consumes the same normalized two-spinor
	carrier as a scalar OFF run, so its transverse fields collapse to the OFF
	values and the bit-identical OFF charge ζ can be reused.  The three raw4
	transverse ζ receipts still pin the current basis independently.  Historical
	raw-charge stamps remain unchanged.  The identity is therefore still
	REQUIRED for a bispinor run even when this one charge stamp ignores it.
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
	_carrier_bispinor = bool(
		cfg.bispinor if carrier_bispinor is None else carrier_bispinor)
	_couple_transverse = bool(
		cfg.bispinor
		and not (int(vertex_mu_L) == 0 and not _carrier_bispinor))
	_ti = dict(transverse_identity or {}) if _couple_transverse else {}
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
	logical_band_stop = int(logical_band_stop)
	left_lo, left_hi = (int(v) for v in band_range_left)
	right_lo, right_hi = (int(v) for v in band_range_right)
	logical_left = [left_lo, min(left_hi, logical_band_stop)]
	logical_right = [right_lo, min(right_hi, logical_band_stop)]
	for name, logical, storage in (
			("left", logical_left, (left_lo, left_hi)),
			("right", logical_right, (right_lo, right_hi))):
		if logical[0] < 0 or logical[1] <= logical[0]:
			raise ValueError(
				f"_zeta_fit_provenance: invalid logical {name} fit range "
				f"{logical} from storage {storage} and "
				f"logical stop {logical_band_stop}.")
	prov = {
		'schema':               _ZETA_PROVENANCE_SCHEMA,
		'n_rmu':                int(meta.n_rmu),
		'band_range_left_logical':  logical_left,
		'band_range_right_logical': logical_right,
		'bispinor':             _carrier_bispinor,
		# The source WFN's spin structure (FILE nspinor, not meta.nspinor:
		# a bispinor run's ζ is fit from the 2-component file ψ, and
		# ``bispinor`` above already separates those stamps).  A ζ fit
		# from an nspinor=2 WFN reused by an nspinor=1 rerun at the same
		# (path, size) fingerprint would evaluate pair densities with the
		# wrong component count.  A stamp MISSING this key predates
		# scalar support, i.e. was fit at nspinor=2 —
		# ``_zeta_reuse_ok``'s legacy table says exactly that.
		'nspinor':              int(meta.nspinor_wfnfile),
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
			if _couple_transverse else 'ridge'),
		'transverse_zeta_rcond': (
			float(cfg.backend.transverse_zeta_rcond)
			if (_couple_transverse
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
	# Additive only for the new normalized carrier.  Historical raw4 and
	# Pauli-reference stamps remain byte-for-byte reusable.
	if carrier_lift is not None:
		prov['bispinor_lift'] = str(carrier_lift)
	# Stamp only the path whose physics changed.  Equal-window charge fits and
	# every transverse fit retain their byte-identical schema-1 provenance;
	# an old asymmetric LR-only stamp lacks this non-legacy key and therefore
	# refits instead of entering the ordered LR+RL serving basis.
	if (int(vertex_mu_L) == 0
			and tuple(band_range_left) != tuple(band_range_right)):
		prov['charge_pair_training_domain'] = 'ordered_lr_plus_rl'
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
                   print_fn=print, *, n_rmu_expected=None,
                   q_irr_is_full_identity=False):
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
			if old.get('schema') != new.get('schema'):
				print_fn(
					f"    [zeta reuse] {zeta_h5_path}: provenance schema "
					f"{old.get('schema')!r} on disk != {new.get('schema')!r} "
					"now.  Legacy stamps do not explicitly identify their "
					"logical fit ranges, so matching storage padding cannot "
					"prove equivalence — refitting.")
				return False
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
			# A q-IBZ REQUEST over the exact identity q table writes the same
			# rows, in the same order, as the historical full-BZ request.  Run20
			# was stamped ``write_ibz_only=false`` before the coupled-head
			# exception was removed; a new run correctly records the requested
			# ``true``.  Accept exactly that old-false/new-true difference only
			# when the caller has proved ``q_irr_full_idx == irr_idx_q ==
			# arange(nq)`` and the parent q table equals the full q table.  This
			# is a diagnosed storage equivalence, not an ignored provenance key.
			if (q_irr_is_full_identity
					and diff == ['write_ibz_only']
					and old.get('write_ibz_only') is False
					and new.get('write_ibz_only') is True):
				print_fn(
					f"    [zeta reuse] {zeta_h5_path}: on-disk provenance "
					"records write_ibz_only=false while this run requests true, "
					"but SymMaps proves the q-IBZ is the full identity q table; "
					"the stored q rows and order are identical, so reuse is "
					"allowed without mutating the artifact.")
				diff.remove('write_ibz_only')
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
		# These feature keys were all introduced within schema 1 and retain
		# their declared legacy meanings when BOTH stamps have the same
		# schema.  Schema-1 -> schema-2 is handled above and fails closed,
		# because no implied value can recover the missing logical band stop.
		# The asymmetric charge pair-domain key is deliberately absent from
		# this table: its old LR-only meaning is not reusable.  Keys and their
		# implied legacy values:
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
		#   nspinor                 2     (2026-08-28: every ζ stamped
		#       before scalar support was fit from an nspinor=2 WFN —
		#       the tree refused nspinor=1 files — so an nspinor=2 rerun
		#       reuses a legacy stamp and an nspinor=1 rerun refits,
		#       naming the key.)
		_LEGACY_KEY_DEFAULTS = {
			'distributed_zeta_solve': 'replicated',
			'transverse_zeta_solve': 'ridge',
			'transverse_zeta_rcond': None,
			'n_rmu_transverse': None,
			'centroids_transverse_md5': None,
			'distributed_lu': None,
			'transverse_solver_kind': None,
			'nspinor': 2,
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


def _transverse_wfn_data(wfn, sym, meta_T, cent_T_idx, cfg, mesh_xy,
                         band_slices, band_chunk_size, k_chunk_size=None):
	"""Sample the current family's packed basis using the same raw-parent loader as charge."""
	from common.wfn_transforms import load_centroids_band_chunked, get_enk_bandrange
	from .wavefunction_bundle import (parent_faces, wavefunctions_face_from_restart,
	                                 build_packed_parent_green_carrier)
	representation = resolve_four_current_representation(cfg.bispinor, cfg.bispinor_gw)
	plan, _, _ = _prepare_parent_wavefunction_plan(
		cfg, meta_T, wfn, band_slices, sym=sym,
		centroid_indices=cent_T_idx, mesh_xy=mesh_xy)
	with timing.section("gw_jax.load_centroid_wfns_current"):
		psi_y, psi_x = load_centroids_band_chunked(
			wfn, sym, meta_T, cent_T_idx, True, mesh_xy,
			band_range=band_slices.full_range, band_chunk_size=int(band_chunk_size),
			k_chunk_size=k_chunk_size, bispinor_lift=representation.current_lift,
			k_domain=sym.parent_k_domain)
	nmu, mun = parent_faces(psi_y, psi_x, mesh_xy=mesh_xy)
	psi_y = psi_x = None
	enk, _ = get_enk_bandrange(wfn, sym, band_slices.full_range,
	                         (band_slices.b1, band_slices.b3), nspinor=4)
	wfns = wavefunctions_face_from_restart(
		None, None, enk_full=enk, slices=band_slices, mesh_xy=mesh_xy)
	carrier = build_packed_parent_green_carrier(
		wfns, nmu, mun, plan=plan, mesh_xy=mesh_xy)
	return dict(meta=meta_T, centroid_indices=cent_T_idx, green_parent=carrier)


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
	log(f"    ζ-fit window DECOUPLED from the band sum: logical physical "
	    f"edge zeta_nband={b4_zeta}; the loaded band carrier ends at "
	    f"b4={band_slices.b4} (any tail above the logical loaded extent is "
	    f"exact-zero mesh padding).  ζ is fitted on left {left} x right "
	    f"{right}.")
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
								  *, meta, log=print):
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

	1. With no explicit narrowing, the fit's STORAGE range tops out at
	   ``max(b4_chi, b4_sigma)`` (== ``b4``, the padded carrier edge).  Its
	   PHYSICAL top is instead the logical count owned by
	   ``Meta.b_id_4_{chi,sigma}_user``.  ``load_centroids_band_chunked``
	   makes the difference exact-zero ψ, so carrier padding is inert and is
	   never called a physical band sum here.
	2. If ``zeta_nband`` narrows the fit BELOW either band sum's top, the
	   consumer above it is running on an EXTRAPOLATED ζ basis.  Reported per
	   consumer, loudly, and named — narrowing is a legitimate request
	   (``zeta_nband`` exists for the BSE's Galerkin capacity bound) but it
	   must never be silent about which sum it undercuts.
	3. The window and the count that set it are logged unconditionally, so
	   "no news" and "a good number" do not look alike.
	"""
	carrier_top = int(band_range_right[1])
	b4_chi, b4_sigma = int(band_slices.b4_chi), int(band_slices.b4_sigma)
	expected_carrier = max(b4_chi, b4_sigma)
	logical_chi = int(meta.b_id_4_chi_user)
	logical_sigma = int(meta.b_id_4_sigma_user)
	expected_logical = max(logical_chi, logical_sigma)

	# ``Meta`` owns the logical-versus-carrier relation.  Assert that the
	# BandSlices handed to this seam still describes that same Meta rather than
	# silently accepting two independently plausible windows.
	if ((logical_chi, logical_sigma) !=
			(min(b4_chi, int(meta.b_id_4_user)),
			 min(b4_sigma, int(meta.b_id_4_user)))):
		raise ValueError(
			"ISDF ζ-fit received BandSlices and Meta with inconsistent logical "
			"χ/Σ tops; the diagnostic cannot establish which carrier rows are "
			"physical.  Build both through the canonical Meta/BandSlices path.")
	if zeta_nband is None and carrier_top != expected_carrier:
		raise ValueError(
			f"ISDF ζ-fit carrier top is {carrier_top} but the loaded band "
			f"carriers reach max(chi={b4_chi}, sigma={b4_sigma}) = "
			f"{expected_carrier}.  The "
			f"interpolation basis MUST span the pair densities of whichever "
			f"consumer reaches higher; a basis fitted to the smaller window "
			f"and used for the larger one is the 0.36 eV / negative-gap "
			f"failure in docs/dev/isdf_basis_adequacy_at_large_nband.md, "
			f"which passed every gate in the suite.  This is a code defect, "
			f"not a deck error.")
	if zeta_nband is not None and carrier_top > expected_logical:
		raise ValueError(
			f"zeta_nband={carrier_top} exceeds the logical loaded band top "
			f"{expected_logical}; [{expected_logical}, {expected_carrier}) is "
			"an exact-zero mesh carrier pad, not physical bands.  Request the "
			"logical edge or omit zeta_nband to follow the padded carrier.")
	physical_top = min(carrier_top, expected_logical)
	source = ("tied" if logical_chi == logical_sigma
	          else ("number_bands_chi" if logical_chi > logical_sigma
	                else "number_bands_sigma"))
	pad_note = (
		f"; loaded carrier top {expected_carrier}, exact-zero inert pad "
		f"[{expected_logical}, {expected_carrier})"
		if expected_carrier > expected_logical else "")
	log(f"    ζ-fit physical window top {physical_top} bands "
	    f"(logical band sums reach chi {logical_chi}, sigma {logical_sigma}; "
	    f"the max is set by {source}{pad_note})"
	    + ("" if physical_top == expected_logical
	       else f", NARROWED from {expected_logical} by deck key "
	            f"zeta_nband={carrier_top}"))
	if zeta_nband is not None:
		for edge, what, key in (
				(logical_chi, "χ0/W band sum", "number_bands_chi"),
				(logical_sigma, "Σ band sum", "number_bands_sigma")):
			if physical_top < edge:
				log(f"    *** zeta_nband={physical_top} is BELOW the {what}'s top "
				    f"({key} → band {edge}).  Bands [{physical_top}, {edge}) "
				    f"of that "
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


@dataclass(frozen=True)
class _ZetaFitContract:
	"""Host-only identities and canonical per-channel zeta reuse verdicts."""

	zeta_h5_path: str
	band_range_left: tuple[int, int]
	band_range_right: tuple[int, int]
	band_norms: object
	zeta_cutoff: float
	zeta_vcoul_cutoff: float
	write_ibz_only_charge: bool
	loader_band_chunk: int
	loader_k_chunk: int
	provenance: str
	reuse_charge: bool
	meta_transverse: object = None
	centroids_transverse: object = None
	transverse_identity: object = None
	write_ibz_only_transverse: bool = False
	zeta_transverse_paths: tuple[str, ...] = ()
	provenance_transverse: tuple[str, ...] = ()
	reuse_transverse: tuple[bool, ...] = ()

	@property
	def reuse(self):
		"""Whether every artifact required by this contract is reusable."""
		return bool(self.reuse_charge and all(self.reuse_transverse))


def _resolve_zeta_fit_contract(
		wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
		*, print_fn=print):
	"""Resolve all zeta identities and reuse verdicts before fit planning.

	Only host metadata and the canonical :func:`_zeta_reuse_ok` owner are used.
	A bispinor contract validates charge and each transverse artifact
	independently, so an interrupted later channel cannot invalidate a completed
	earlier one.  The derived ``reuse`` property remains the all-artifact bundle
	verdict.  No wavefunction sampling, fit planner, or zeta data load occurs
	here.
	"""
	zeta_edge = resolve_zeta_fit_edge(
		band_slices, getattr(cfg, "zeta_nband", None))
	band_range_left, band_range_right = zeta_fit_band_ranges(
		band_slices, zeta_edge, log=print_fn)
	logical_band_stop = (
		int(zeta_edge) if zeta_edge is not None
		else int(getattr(meta, "b_id_4_user", 0) or band_slices.b4))
	assert_isdf_window_is_the_max(
		band_slices, band_range_right, zeta_edge, meta=meta, log=print_fn)
	check_zeta_fit_windows(
		getattr(wfn, "energies", None), band_range_left, band_range_right,
		zeta_edge, logical_band_stop,
		log=print_fn)

	zeta_h5_path = os.path.join(tmp_dir, "zeta_q.h5")
	_check_zeta_h5_matches_basis(
		zeta_h5_path, int(meta.n_rmu), print_fn, fft_grid=meta.fft_grid)
	band_norms = getattr(wfn, "band_norms", None)

	write_ibz_only_charge = True

	def _resolve_cutoff(val, label, hi):
		if val is None:
			return float(wfn.ecutwfc)
		v = float(val)
		if v > hi + 1e-9:
			raise ValueError(
				f"{label} = {v} Ry exceeds ecutrho = {hi} Ry "
				"(the FFT grid can't represent G's past ecutrho).")
		return v

	ecutrho = float(wfn.ecutrho)
	zeta_vcoul_cutoff = _resolve_cutoff(
		cfg.head.bare_coulomb_cutoff, "bare_coulomb_cutoff", ecutrho)
	zeta_cutoff = _resolve_cutoff(
		cfg.head.zeta_cutoff, "zeta_cutoff", ecutrho)
	if zeta_vcoul_cutoff > zeta_cutoff + 1e-9:
		raise ValueError(
			f"bare_coulomb_cutoff = {zeta_vcoul_cutoff} Ry > "
			f"zeta_cutoff = {zeta_cutoff} Ry.  The V_q kernel reads "
			f"ζ̃(q+G) at every G inside its sphere; ζ must be stored "
			f"at least as wide.  Increase zeta_cutoff (≤ ecutrho = "
			f"{ecutrho} Ry) or lower bare_coulomb_cutoff.")
	print_fn(
		f"    cutoffs: zeta = {zeta_cutoff:.1f} Ry, "
		f"bare-Coulomb = {zeta_vcoul_cutoff:.1f} Ry  "
		f"(ecutwfc={float(wfn.ecutwfc):.1f}, ecutrho={ecutrho:.1f})")

	meta_transverse = None
	centroids_transverse = None
	transverse_identity = None
	write_ibz_only_transverse = False
	transverse_paths = ()
	if cfg.bispinor:
		if not getattr(cfg.paths, "centroids_file_current", None):
			raise ValueError(
				"Bispinor calculation requires centroids_file_current in "
				"cohsex.in (set it to a current-density kmeans output).")
		from file_io.centroids import load_centroids as _load_cent_pf
		from isdf.core import _resolve_solver_kind_transverse
		from runtime.padding import padded_mu_extent
		_, cent_T_np, n_rmu_T = _load_cent_pf(
			cfg.paths.centroids_file_current, meta.fft_grid)
		# Keep the pre-fit contract on the host.  The centroid table becomes a
		# device array only after the canonical all-channel reuse verdict says
		# that a fit or downstream Sigma sampling is actually required.
		centroids_transverse = np.asarray(cent_T_np, dtype=np.int32)
		from common.centroid_basis import PackedCentroidBasis
		basis_T = PackedCentroidBasis.build(
			centroids_transverse, sym, meta.fft_grid, mesh_xy)
		meta_transverse = replace(
			meta, n_rmu=int(n_rmu_T), nspinor=4, npol=4, mu_basis=basis_T,
			n_rmu_padded=basis_T.n_packed)
		solver_kind_T = _resolve_solver_kind_transverse(
			mesh_xy, cfg.backend.distributed_lu,
			n_rmu_logical=meta_transverse.mu_solve_extent,
			transverse_zeta_solve=cfg.backend.transverse_zeta_solve)
		meta_transverse.sys_dim = meta.sys_dim
		meta_transverse.bispinor = True
		transverse_identity = {
			"n_rmu": int(n_rmu_T),
			"centroids_md5": _centroid_table_md5(cent_T_np),
			"distributed_lu": str(cfg.backend.distributed_lu).strip().lower(),
			"solver_kind": str(solver_kind_T),
		}
		write_ibz_only_transverse = write_ibz_only_charge
		transverse_paths = tuple(
			os.path.join(tmp_dir, f"zeta_q_mu{mu_L}.h5")
			for mu_L in (1, 2, 3))

	representation = resolve_four_current_representation(
		cfg.bispinor, cfg.bispinor_gw)
	# One carrier for both shipped bispinor_gw values (the raw kinetic
	# balance lift), so the zeta provenance names no alternate lift.  The
	# two comparison carriers that did were retired from the deck grammar
	# on 2026-09-01 (gw_config._RETIRED_BISPINOR_GW_MODES).
	_normalized_charge_lift = None
	_normalized_current_lift = None
	provenance = _zeta_fit_provenance(
		wfn=wfn, meta=meta, cfg=cfg,
		band_range_left=band_range_left,
		band_range_right=band_range_right,
		logical_band_stop=logical_band_stop,
		zeta_cutoff=zeta_cutoff,
		zeta_vcoul_cutoff=zeta_vcoul_cutoff,
		write_ibz_only=write_ibz_only_charge,
		band_norms=band_norms,
		carrier_bispinor=bool(int(meta.nspinor) == 4),
		carrier_lift=_normalized_charge_lift,
		vertex_mu_L=0, transverse_identity=transverse_identity)
	provenance_transverse = tuple(
		_zeta_fit_provenance(
			wfn=wfn, meta=meta_transverse, cfg=cfg,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			logical_band_stop=logical_band_stop,
			zeta_cutoff=zeta_cutoff,
			zeta_vcoul_cutoff=zeta_vcoul_cutoff,
			write_ibz_only=write_ibz_only_transverse,
			band_norms=band_norms, carrier_bispinor=True,
			carrier_lift=_normalized_current_lift,
			vertex_mu_L=mu_L, transverse_identity=transverse_identity)
		for mu_L in ((1, 2, 3) if cfg.bispinor else ()))
	q_irr_identity = bool(sym.q_irr_is_full_identity)
	reuse_charge = _zeta_reuse_ok(
		zeta_h5_path, provenance, centroid_indices,
		print_fn=print_fn, n_rmu_expected=int(meta.n_rmu),
		q_irr_is_full_identity=q_irr_identity)
	reuse_transverse = tuple(
		bool(_zeta_reuse_ok(
			path_T, provenance_T, centroids_transverse,
			print_fn=print_fn,
			n_rmu_expected=int(meta_transverse.n_rmu),
			q_irr_is_full_identity=q_irr_identity))
		for path_T, provenance_T in
		zip(transverse_paths, provenance_transverse))

	loader_band_chunk = (
		int(cfg.memory.band_chunk_size)
		if int(cfg.memory.band_chunk_size) > 0 else 64)
	# Reuse skips the full fit planner by design, but it still re-samples the
	# WFN for downstream Sigma.  Resolve the same pure Stage-A geometry here so
	# a valid zeta artifact cannot turn k streaming off by setting ``chunks`` to
	# None.  The geometry owner and common loader both route physical band-tile
	# rounding through runtime.padding.round_up; nothing allocates or compiles.
	from gw.gflat_memory_model import centroid_fft_tile_geometry
	from common.wfn_layout import band_sphere_spec
	from runtime.padding import spec_divisor
	_loader_band_request = min(
		int(band_slices.full_range[1] - band_slices.full_range[0]),
		int(loader_band_chunk))
	_loader_p_band = spec_divisor(
		mesh_xy, band_sphere_spec(), axis=1)
	loader_k_chunk, _ = centroid_fft_tile_geometry(
		nk=int(meta.nk_tot), band_chunk=_loader_band_request,
		p_band=_loader_p_band)
	return _ZetaFitContract(
		zeta_h5_path=zeta_h5_path,
		band_range_left=band_range_left,
		band_range_right=band_range_right,
		band_norms=band_norms,
		zeta_cutoff=zeta_cutoff,
		zeta_vcoul_cutoff=zeta_vcoul_cutoff,
		write_ibz_only_charge=write_ibz_only_charge,
		loader_band_chunk=loader_band_chunk,
		loader_k_chunk=loader_k_chunk,
		provenance=provenance,
		reuse_charge=bool(reuse_charge),
		meta_transverse=meta_transverse,
		centroids_transverse=centroids_transverse,
		transverse_identity=transverse_identity,
		write_ibz_only_transverse=write_ibz_only_transverse,
		zeta_transverse_paths=transverse_paths,
		provenance_transverse=provenance_transverse,
		reuse_transverse=reuse_transverse)


def _plan_gflat_chunks_for_channel(
		*, meta, cfg, band_slices, mesh_xy, is_bispinor, n_q_selected,
		face_current_vertex=False, parent_route=None, print_fn=print):
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
	``centroid_k_chunk`` / ``chunk_r`` / ``q_chunk`` / ``gflat_chunk_size`` /
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
	n_q_selected = int(n_q_selected)
	if not 1 <= n_q_selected <= int(meta.nk_tot):
		raise ValueError(
			"selected zeta q rows must be in [1, full-zone K], got "
			f"Q={n_q_selected}, K={int(meta.nk_tot)}")
	_ngkmax = int(getattr(meta, 'ngkmax', 0)) or int(0.06 * meta.n_rtot)
	gflat_plan = plan_gflat_chunks(
		meta=meta, mesh_xy=mesh_xy,
		nb_total=nb_total,
		face_nb_total=int(band_slices.b4 - band_slices.b0),
		fit_nb_total=_zeta_fit_nb,
		ngkmax=_ngkmax,
		n_q_disk=n_q_selected,
		n_q_ibz=n_q_selected,
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
		face_current_vertex=bool(face_current_vertex),
		parent_route=parent_route,
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
		'centroid_k_chunk': int(gflat_plan.centroid_k_chunk),
		'chunk_r': int(gflat_plan.r_chunk),
		'q_chunk': int(gflat_plan.q_chunk),
		'gflat_chunk_size': int(gflat_plan.gflat_chunk_size),
		'cache_psi_r': bool(gflat_plan.cache_psi_r),
		'cache_face_y_blocks': bool(gflat_plan.cache_face_y_blocks),
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


def _gate_fresh_zeta_rank_findings(
		finding_context, *, transverse=False, print_fn=print):
	"""Dispose deferred rank findings before a fresh ζ channel is stamped.

	The ζ truncations execute inside jitted kernels and therefore record through
	the incumbent ``spectral_closure`` and ``rank_criterion`` services.  Every
	fresh charge or transverse writer must cross this host seam immediately
	after its fit, before provenance makes that artifact reusable.  Accepted
	restart artifacts never call this function.
	"""
	_sc_mode = spectral_closure.resolve_mode(
		os.environ.get(spectral_closure.MODE_ENV))
	# Read findings before the disposition clears them, so the clean/fired
	# banner below cannot contradict the service result.
	_sc_fired = bool(spectral_closure.pending())
	spectral_closure.raise_if_pending(
		finding_context, mode=_sc_mode, log=print_fn)
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

	_rp_mode = rank_criterion.resolve_policy_mode(
		os.environ.get(rank_criterion.POLICY_MODE_ENV))
	_rp_fired = bool(rank_criterion.pending())
	rank_criterion.raise_if_pending(
		finding_context, mode=_rp_mode, log=print_fn)
	if _rp_mode == "off":
		print_fn("    ζ rank-cut certification: gate is OFF "
		         f"({rank_criterion.POLICY_MODE_ENV}=off) — NOT CHECKED, "
		         f"which is an absence and not a pass.")
	elif _rp_fired:
		print_fn(f"    ζ rank-cut certification: FIRED above "
		         f"(mode={_rp_mode}).  The cut bound outside the regime any "
		         f"measurement certifies; the numbers are in the "
		         f"[rank-policy] lines.")
	elif transverse:
		print_fn(f"    ζ rank-cut certification: ARMED (mode={_rp_mode}, "
		         f"transverse κ is UNCERTIFIED, discarded-weight ceiling "
		         f"{rank_criterion.DISCARDED_WEIGHT_MAX:.1e}) and SILENT — "
		         f"either the cut bound nothing, or its discarded weight "
		         f"remained inside the certified regime.")
	else:
		# Preserve the established charge-channel message verbatim.
		print_fn(f"    ζ rank-cut certification: ARMED (mode={_rp_mode}, "
		         f"certified κ ceiling "
		         f"{rank_criterion.KAPPA_CERTIFIED_GRAM:.1e} for the charge "
		         f"Gram, discarded-weight ceiling "
		         f"{rank_criterion.DISCARDED_WEIGHT_MAX:.1e}) and SILENT — "
		         f"either the cut bound nothing, or it bound inside the "
		         f"certified regime.  The transverse channel is "
		         f"UNCERTIFIED and can only raise the weight finding.")


class _CoupledMu123ZqCoordinator:
	"""Host scheduler for the experimental three-current Zq transaction.

	Channel setup runs one-at-a-time.  For each r chunk, μ=1 builds one shared
	``[3,q,mu,r]`` stack: raw Zq on the general route, or solved zeta when the
	three batch-reshard solves can be flattened into one transaction.  μ=1,2,3
	then consume its slices in the accepted accumulate order.  The stack is
	released immediately after μ=3 finishes that chunk.  Final G-flat writes
	and provenance also retain the accepted μ=1→2→3 order.
	"""
	_CHANNELS = (1, 2, 3)

	def __init__(self):
		self._cv = threading.Condition()
		self._prepared = set()
		self._release_prepared = False
		self._aborted = None
		self._chunk = 0
		self._arrived = set()
		self._chunk_stack = None
		self._turn = 1
		self._solve_inputs = {}
		self._stacked_solve_inputs = None
		self._stacked_factor_ready = False
		self._final_ready = set()
		self._final_turn = 1

	def _raise_if_aborted(self):
		if self._aborted is not None:
			raise RuntimeError(
				"coupled mu123 Zq transaction aborted") from self._aborted

	def abort(self, exc):
		with self._cv:
			if self._aborted is None:
				self._aborted = exc
			self._cv.notify_all()

	def channel_prepared(self, mu, solve_inputs=None):
		mu = int(mu)
		with self._cv:
			if mu not in self._CHANNELS or mu in self._prepared:
				raise ValueError(f"invalid/duplicate prepared channel mu={mu}")
			self._prepared.add(mu)
			self._solve_inputs[mu] = solve_inputs
			self._cv.notify_all()
			while not self._release_prepared:
				self._raise_if_aborted()
				self._cv.wait()
			self._raise_if_aborted()

	def wait_channel_prepared(self, mu):
		with self._cv:
			while int(mu) not in self._prepared:
				self._raise_if_aborted()
				self._cv.wait()

	def release_channels(self):
		with self._cv:
			if self._prepared != set(self._CHANNELS):
				raise RuntimeError(
					"cannot release coupled channels before all are prepared")
			ordered = tuple(self._solve_inputs[mu] for mu in self._CHANNELS)
		if any(item is not None for item in ordered):
			if not all(item is not None for item in ordered):
				raise RuntimeError(
					"coupled channels disagreed on the stacked-solve route")
			factors, traces = zip(*ordered)
			stacked_factor = jnp.concatenate(factors, axis=0)
			stacked_trace = jnp.concatenate(traces, axis=0)
			jax.block_until_ready((stacked_factor, stacked_trace))
			stacked_inputs = (stacked_factor, stacked_trace)
		else:
			stacked_inputs = None
		with self._cv:
			self._stacked_solve_inputs = stacked_inputs
			# The concatenated carrier is now the sole factor/trace owner for
			# the stacked route.  Retaining the three registration tuples would
			# keep an avoidable second copy alive through every r chunk.
			self._solve_inputs.clear()
			self._release_prepared = True
			self._cv.notify_all()

	def stacked_solve_inputs(self, *, factorize=None):
		with self._cv:
			if not self._release_prepared:
				raise RuntimeError(
					"stacked solve inputs requested before channel release")
			if self._stacked_solve_inputs is None:
				raise RuntimeError(
					"coupled transaction did not register stacked solve inputs")
			if factorize is not None and not self._stacked_factor_ready:
				self._stacked_solve_inputs = factorize(*self._stacked_solve_inputs)
				self._stacked_factor_ready = True
			return self._stacked_solve_inputs

	def _acquire_channel_stack(self, mu, chunk_idx, builder):
		mu = int(mu)
		chunk_idx = int(chunk_idx)
		build_here = False
		with self._cv:
			while chunk_idx != self._chunk:
				self._raise_if_aborted()
				self._cv.wait()
			if mu in self._arrived:
				raise RuntimeError(
					f"duplicate coupled-Zq arrival mu={mu}, chunk={chunk_idx}")
			self._arrived.add(mu)
			self._cv.notify_all()
			while self._arrived != set(self._CHANNELS):
				self._raise_if_aborted()
				self._cv.wait()
			if mu == 1 and self._chunk_stack is None:
				build_here = True
			else:
				while self._chunk_stack is None:
					self._raise_if_aborted()
					self._cv.wait()

		if build_here:
			try:
				z_stack = builder()
				z_stack.block_until_ready()
			except BaseException as exc:
				self.abort(exc)
				raise
			with self._cv:
				self._chunk_stack = z_stack
				self._cv.notify_all()

		with self._cv:
			while self._turn != mu:
				self._raise_if_aborted()
				self._cv.wait()
			self._raise_if_aborted()
			return self._chunk_stack[mu - 1]

	def acquire_channel_Z_q(self, mu, chunk_idx, builder):
		return self._acquire_channel_stack(mu, chunk_idx, builder)

	def acquire_channel_zeta(self, mu, chunk_idx, builder):
		return self._acquire_channel_stack(mu, chunk_idx, builder)

	def finish_chunk(self, mu, chunk_idx):
		mu = int(mu)
		with self._cv:
			if int(chunk_idx) != self._chunk or mu != self._turn:
				raise RuntimeError(
					f"out-of-order coupled chunk finish mu={mu}, "
					f"chunk={chunk_idx}, expected mu={self._turn}, "
					f"chunk={self._chunk}")
			if mu < 3:
				self._turn += 1
			else:
				self._chunk_stack = None
				self._arrived.clear()
				self._turn = 1
				self._chunk += 1
			self._cv.notify_all()

	def wait_finalize(self, mu):
		mu = int(mu)
		with self._cv:
			if mu not in self._CHANNELS or mu in self._final_ready:
				raise ValueError(f"invalid/duplicate final-ready channel mu={mu}")
			self._final_ready.add(mu)
			self._cv.notify_all()
			while (self._final_ready != set(self._CHANNELS)
			       or self._final_turn != mu):
				self._raise_if_aborted()
				self._cv.wait()
			self._raise_if_aborted()

	def finish_channel(self, mu):
		with self._cv:
			if self._final_turn != int(mu):
				raise RuntimeError(
					f"out-of-order coupled channel finish mu={mu}, "
					f"expected {self._final_turn}")
			self._final_turn += 1
			self._cv.notify_all()


def _select_coupled_mu123_route(*, requested_route, base_hwm_bytes,
		budget_bytes, local_delta_bytes, distributed_delta_bytes,
		local_capacity_ok=True):
	"""Choose the fastest coupled transverse schedule that fits device HBM.

	This policy is private to the three current-density ζ fits.  The public
	``distrib_la_batched_route`` still governs every other consumer.  An
	explicit ``batch_reshard`` request is never silently changed to the
	distributed service; if its coupled live set does not fit, the incumbent
	sequential schedule retains that explicit per-channel route.
	"""
	route = str(requested_route).strip().lower()
	if route not in ("auto", "batch_reshard"):
		raise ValueError(
			f"unsupported distrib_la_batched_route={requested_route!r}")
	base = float(base_hwm_bytes)
	budget = float(budget_bytes)
	local_delta = float(local_delta_bytes)
	distributed_delta = float(distributed_delta_bytes)
	if bool(local_capacity_ok) and base + local_delta <= budget:
		return True, "batch_reshard", local_delta
	if route == "auto" and base + distributed_delta <= budget:
		return True, "auto", distributed_delta
	return False, route, None


def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
             chunks, print_fn=print,
             zeta_contract=None,
             k_unfold_plan=None, psi_nmu_parent=None, psi_mun_parent=None):
	"""Fit missing charge/current ζ channels from their typed parents, preserving independent reuse."""
	from gw.isdf_fitting import fit_zeta_to_h5
	from common.gamma_matrices import set_gamma_contract_mode
	representation = resolve_four_current_representation(
		cfg.bispinor, cfg.bispinor_gw)
	# Honour cohsex.in ``gamma_contract_mode`` for the γ̃·γ̃ kernel
	# inside the monolithic pair pipeline.  Mode is module-level (the
	# γ̃ contract sits inside shard_map bodies so threading a kwarg
	# through every call would be churn for no benefit).
	set_gamma_contract_mode(cfg.backend.gamma_contract_mode)

	if zeta_contract is None:
		zeta_contract = _resolve_zeta_fit_contract(
			wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices,
			tmp_dir, print_fn=print_fn)
	band_range_left = zeta_contract.band_range_left
	band_range_right = zeta_contract.band_range_right
	zeta_h5_path = zeta_contract.zeta_h5_path
	_band_norms = zeta_contract.band_norms
	_zeta_cutoff = zeta_contract.zeta_cutoff
	_zeta_vcoul_cutoff = zeta_contract.zeta_vcoul_cutoff
	_write_ibz_only_charge = zeta_contract.write_ibz_only_charge
	_reuse_charge = bool(zeta_contract.reuse_charge)
	_reuse_T = tuple(bool(value) for value in zeta_contract.reuse_transverse)

	# Chunk sizes (band_chunk / chunk_r / q_chunk / gflat_chunk_size) were
	# picked once by ``plan_gflat_chunks`` in the caller and live in
	# ``chunks``; fit_zeta is a pure consumer.
	mem_est = chunks.get('memory_estimate', {}) if chunks is not None else {}
	if chunks is not None:
		print_fn(f"\n  Chunked ISDF fitting:")
		print_fn(f"    Band chunks: {chunks['band_chunk']}")
		print_fn(f"    R chunks:    {chunks['chunk_r']} (contiguous r-space)")
		print_fn(f"    Q chunks:    {chunks['q_chunk']}")
		print_fn("    ψ(r) source: " + (
			"hoisted all-band cache" if chunks.get('cache_psi_r', True)
			else "streamed band-chunk FFT"))
		if chunks.get('gflat_chunk_size') is not None:
			print_fn(f"    GFlat cs:    {chunks['gflat_chunk_size']}")
		print_fn(f"    Zeta output: {zeta_h5_path}")
	if zeta_contract.reuse:
		print_fn("")
		print_fn("  " + "=" * 68)
		print_fn(f"  REUSING the existing ζ at {zeta_h5_path} — FIT SKIPPED.")
		if cfg.bispinor:
			print_fn("  ...and the three transverse ζ at "
			         "zeta_q_mu{1,2,3}.h5.")
		print_fn("  isdf_header/zeta_is_done is True, the centroid table")
		print_fn("  matches, and fit_provenance is identical to this run's")
		print_fn("  inputs (band windows, cutoffs, solver knobs, source WFN).")
		print_fn("  Set LORRAX_FORCE_REFIT=1 to refit unconditionally.")
		print_fn("  " + "=" * 68)
		print_fn("")
		if not cfg.bispinor:
			return zeta_h5_path, mem_est, None
		print_fn(
			f"  [bispinor] re-sampling ψ at the "
			f"{int(zeta_contract.meta_transverse.n_rmu)} transverse "
			"centroids for σ^B (the ζ_T fit is skipped, but Σ^B still "
			"needs ψ(r_{μ_T})).")
		return zeta_h5_path, mem_est, _transverse_wfn_data(
			wfn, sym, zeta_contract.meta_transverse,
			zeta_contract.centroids_transverse, cfg, mesh_xy, band_slices,
			zeta_contract.loader_band_chunk,
			k_chunk_size=zeta_contract.loader_k_chunk)
	# Any missing bispinor channel consumes the transverse identity already
	# resolved by the per-channel contract.  Only the fit-specific chunk plan
	# remains here.
	_meta_T = zeta_contract.meta_transverse
	_cent_T_idx = zeta_contract.centroids_transverse
	_transverse_identity = zeta_contract.transverse_identity
	_chunks_T = None
	_gflat_plan_T = None
	_write_ibz_only_transverse = zeta_contract.write_ibz_only_transverse
	if cfg.bispinor:
		if len(_reuse_T) != 3:
			raise AssertionError(
				"a bispinor zeta contract must carry three transverse "
				"reuse verdicts")
		_cent_T_idx = jnp.asarray(
			zeta_contract.centroids_transverse, dtype=jnp.int32)
		# Chunk-plan the TRANSVERSE channel SEPARATELY from the charge
		# ``chunks`` this function was handed.  μ_T is typically ≈ μ_C/3,
		# and reusing the charge-sized plan unchanged for all three ζ_T
		# fits is exactly the register row this closes: "three ζ_T fits
		# inherit the CHARGE chunk plan (μ_T≈μ_C/3): ~3x extra r-chunks,
		# ~2.7 GB/rank avoidable gather".  ONE call here, after reuse was
		# definitively declined and ahead of the μ_L fit loop; all three
		# Lorentz components share this one transverse-sized plan.
		if not all(_reuse_T):
			_n_q_selected_T = (
				int(np.asarray(sym.q_irr_full_idx).shape[0])
				if _write_ibz_only_transverse else int(_meta_T.nk_tot))
			_chunks_T, _gflat_plan_T = _plan_gflat_chunks_for_channel(
				meta=_meta_T, cfg=cfg, band_slices=band_slices,
				mesh_xy=mesh_xy, is_bispinor=True,
				n_q_selected=_n_q_selected_T,
				parent_route=dict(n_parent=int(np.asarray(sym.kirr_fullids).size),
				                  parents_only=True),
				face_current_vertex=True, print_fn=print_fn)
	_coupled_mu123_enabled = False
	_transverse_batched_route = str(
		cfg.backend.distrib_la_batched_route).strip().lower()
	if cfg.bispinor and any(_reuse_T) and not all(_reuse_T):
		print_fn(
			"  [bispinor] partial transverse ζ reuse: fitting only missing "
			"channels on the sequential schedule.")
	if (cfg.bispinor and not any(_reuse_T)
			and bool(_chunks_T.get('cache_face_y_blocks', False))):
		from gw.gflat_memory_model import (
			_batch_reshard_operand_floor_bytes,
			_coupled_route_projected_hwm_bytes,
			_coupled_mu123_zq_incremental_bytes)
		from common.gpu_utils import get_cpu_memory_total
		_p_x = int(mesh_xy.shape['x'])
		_p_y = int(mesh_xy.shape['y'])
		_delta_args = dict(
			nk=int(_meta_T.nk_tot), nq=int(_meta_T.nk_tot),
			ns=int(_meta_T.nspinor), mu=int(_meta_T.n_rmu_padded),
			face_nb=int(band_slices.b4 - band_slices.b0),
			r_chunk=int(_chunks_T['chunk_r']), p_x=_p_x, p_y=_p_y,
			ngkmax=(int(getattr(_meta_T, 'ngkmax', 0))
			          or int(0.06 * _meta_T.n_rtot)),
			n_rtot=int(_meta_T.n_rtot),
			cache_psi_r=bool(_chunks_T.get('cache_psi_r', True)),
			host_spill_gflat=True)
		# The shared Zq transform is coupled, but the three real transverse
		# systems retain their accepted 36-q solve boundaries.  Flattening them
		# to 108 q exposed input-sensitive ~1e-9 arithmetic drift on CrI3 while
		# saving only a small solve dispatch inside a transform-dominated chunk.
		_local_delta = _coupled_mu123_zq_incremental_bytes(
			**_delta_args, stack_three_solves=False)
		_distributed_delta = _coupled_mu123_zq_incremental_bytes(
			**_delta_args, stack_three_solves=False)
		from runtime.padding import mesh_divisor
		_p_xy = mesh_divisor(mesh_xy)
		_mu_T = int(_meta_T.n_rmu_padded)
		_local_operand_floor = _batch_reshard_operand_floor_bytes(
			batch=int(_meta_T.nk_tot), mu=_mu_T,
			nrhs=int(_chunks_T['chunk_r']), processes=_p_xy)
		_local_devices = jax.local_devices()
		_device_kind = (
			str(_local_devices[0].device_kind).upper()
			if _local_devices else "")
		_certified_local_backend = (
			jax.default_backend() in ('gpu', 'cuda')
			and 'A100' in _device_kind
			and _p_x == _p_y and _p_xy in (4, 16))
		# The deck/planner budget is rank-invariant and no larger than the
		# allocator pool.  Using it for the measured 50% operand ceiling is a
		# conservative static dispatch; process-local memory_stats could differ
		# transiently and must not choose different collective routes by rank.
		_allocator_limit = float(_gflat_plan_T.budget_bytes)
		_local_capacity_ok = (
			_certified_local_backend and _mu_T <= 16_384
			and _local_operand_floor <= 0.50 * _allocator_limit)
		try:
			_slurm_nodes = max(1, int(os.environ.get('SLURM_NNODES', '1')))
		except ValueError:
			_slurm_nodes = 1
		_ranks_per_node = (
			int(jax.process_count()) + _slurm_nodes - 1) // _slurm_nodes
		_host_total_gb = get_cpu_memory_total()
		_host_required_node = (
			_local_delta['three_host_gflat_outputs'] * _ranks_per_node)
		_host_spill_ok = (
			_host_total_gb is not None
			and _host_required_node <= 0.35 * _host_total_gb * 1024**3)
		_effective_device_budget = (
			float(_gflat_plan_T.budget_bytes)
			* float(_gflat_plan_T.target_utilization))
		_local_projected_hwm = _coupled_route_projected_hwm_bytes(
			base_hwm=_gflat_plan_T.hwm_bytes,
			persistent=_gflat_plan_T.persistent_bytes,
			coupled_delta=_local_delta['total'],
			solve_operand_floor=_local_operand_floor)
		_local_budget_delta = max(
			0.0, _local_projected_hwm - float(_gflat_plan_T.hwm_bytes))
		if _host_spill_ok:
			(_coupled_mu123_enabled, _transverse_batched_route,
			 _selected_delta_bytes) = _select_coupled_mu123_route(
				requested_route=_transverse_batched_route,
				base_hwm_bytes=_gflat_plan_T.hwm_bytes,
				budget_bytes=_effective_device_budget,
				local_delta_bytes=_local_budget_delta,
				distributed_delta_bytes=_distributed_delta['total'],
				local_capacity_ok=_local_capacity_ok)
		else:
			_selected_delta_bytes = None
		if _coupled_mu123_enabled:
			_coupled_delta = (
				_local_delta if _transverse_batched_route == 'batch_reshard'
				else _distributed_delta)
			_coupled_projected = (
				float(_gflat_plan_T.hwm_bytes) + _selected_delta_bytes)
			print_fn(
				f"  [bispinor] coupled μ_L=1,2,3 schedule: "
				f"solve_route={_transverse_batched_route}, "
				f"device increment {_selected_delta_bytes / 1e9:.2f} GB, "
				f"projected HWM {_coupled_projected / 1e9:.2f} GB; "
				f"three G-flat outputs use "
				f"{_coupled_delta['three_host_gflat_outputs'] / 1e9:.2f} "
				f"GB host/rank, {_host_required_node / 1e9:.2f} GB/node.")
		else:
			_reason = (
				f"host spill {_host_required_node / 1e9:.2f} GB/node exceeds "
				"the 35% host-RAM cap"
				if not _host_spill_ok else
				"coupled live set exceeds the fragmentation-safe device budget")
			print_fn(
				f"  [bispinor] {_reason}; using the sequential capacity "
				"fallback.")
	# Fresh writers and provenance stamps consume exactly the identity that
	# was tested before planning; there is no second reconstruction here.
	_provenance = zeta_contract.provenance

	def _provenance_T(mu_L):
		return zeta_contract.provenance_transverse[int(mu_L) - 1]

	_zeta_T_paths = {
		mu_L: path for mu_L, path in
		enumerate(zeta_contract.zeta_transverse_paths, start=1)
	}
	if not _reuse_charge and chunks is None:
		raise AssertionError(
			"a non-reusable charge zeta reached fit_zeta without the canonical "
			"G-flat chunk plan")
	peak_bytes = 0
	if _reuse_charge:
		print_fn(f"  [zeta reuse] charge ζ accepted at {zeta_h5_path}; "
		         "charge fit skipped independently.")
	else:
		with timing.section("gw_jax.zeta_fit_chunked"), \
		     jax_profile.trace_section("zeta_fit"):
			peak_bytes = fit_zeta_to_h5(
				wfn=wfn, sym=sym, meta=meta,
				centroid_indices=centroid_indices, mesh_xy=mesh_xy,
				chunk_r=chunks['chunk_r'], output_file=zeta_h5_path,
				band_chunk_size=chunks['band_chunk'],
				q_chunk_size=chunks['q_chunk'],
				bispinor=bool(int(meta.nspinor) == 4),
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
				cache_psi_r=bool(chunks.get('cache_psi_r', True)),
				cache_face_y_blocks=bool(
					chunks.get('cache_face_y_blocks', False)),
				write_ibz_only=_write_ibz_only_charge,
				zeta_cutoff_ry=_zeta_cutoff,
				print_fn=print_fn,
				bispinor_lift=(representation.charge_lift or "raw"),
				k_unfold_plan=k_unfold_plan,
				psi_nmu_parent=psi_nmu_parent,
				psi_mun_parent=psi_mun_parent,
			)

	# Device kernels can only record closure/certification findings.  The
	# shared host seam must dispose them after this fresh writer and before its
	# provenance stamp makes the artifact reusable.
	if not _reuse_charge:
		_gate_fresh_zeta_rank_findings(
			"the ζ fit's rank truncation", print_fn=print_fn)

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
	if not _reuse_charge and _trunc:
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
	elif not _reuse_charge and jax.process_index() == 0:
		try:
			from file_io.isdf_header import stamp_fit_provenance
			stamp_fit_provenance(zeta_h5_path, _provenance)
		except Exception as exc:
			# Non-fatal: the ζ itself is fine, it just won't be reusable.
			print_fn(f"    [zeta provenance] not stamped ({exc}); this ζ "
			         f"will be refit on the next run.")
	if not _reuse_charge:
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
			         f"run's host RSS — use LORRAX_DEBUG_PRINT=1 for that]")

	# Default: no transverse-channel ψ to surface to the caller.
	transverse_wfn_data = None

	# ── Bispinor: fit ζ^{μ_L=1,2,3} on the current-density centroid set ──
	# Same kernel as the charge channel, swapping in the γ̃^i vertex.  The
	# automatic coupled schedule shares the expensive face transform; its
	# capacity fallback makes three sequential calls.  Output paths follow the
	# convention zeta_q_mu{1,2,3}.h5 next to zeta_q.h5.
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
		print_fn(f"\n  [bispinor] resolving ζ^{{μ_L=1,2,3}} on current-density "
		         f"centroids: {cfg.paths.centroids_file_current}")
		# ``meta_T``, the centroid table and its μ padding were built ONCE
		# in the bispinor pre-flight (they are inputs to the ζ-reuse
		# decision, which runs before the charge fit).
		meta_curr = _meta_T
		cents_curr_idx = _cent_T_idx
		transverse_wfn_data = _transverse_wfn_data(
			wfn, sym, meta_curr, cents_curr_idx, cfg, mesh_xy,
			band_slices,
			(_chunks_T['band_chunk'] if _chunks_T is not None
			 else zeta_contract.loader_band_chunk),
			k_chunk_size=(
				_chunks_T['centroid_k_chunk']
				if _chunks_T is not None
				else zeta_contract.loader_k_chunk))
		parent_T = transverse_wfn_data['green_parent']

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

		def _drain_coupled_rank_findings(mu_L, stage):
			# The services are process-global.  The coupled schedule serializes
			# this callback with the corresponding preparation/solve so a
			# deferred finding can never be attributed to the next current.
			# Stay silent on the overwhelmingly common empty path; the accepted
			# per-channel gate below still emits its canonical final banner.
			if spectral_closure.pending() or rank_criterion.pending():
				_gate_fresh_zeta_rank_findings(
					f"the μ_L={mu_L} transverse ζ fit's {stage}",
					transverse=True, print_fn=print_fn)

		def _fit_transverse_channel(mu_L, coordinator=None):
			zeta_mu_path = _zeta_T_paths[mu_L]
			if _reuse_T[mu_L - 1]:
				print_fn(f"  [zeta reuse] μ_L={mu_L} accepted at "
				         f"{zeta_mu_path}; fit skipped independently.")
				return
			print_fn(f"  [bispinor] μ_L={mu_L} → {zeta_mu_path}")
			with timing.section(f"gw_jax.zeta_fit_chunked_mu{mu_L}"), \
			     jax_profile.trace_section(f"zeta_fit_mu{mu_L}"):
				fit_zeta_to_h5(
					wfn=wfn, sym=sym, meta=meta_curr,
					centroid_indices=cents_curr_idx,
					mesh_xy=mesh_xy,
					chunk_r=_chunks_T['chunk_r'], output_file=zeta_mu_path,
					k_unfold_plan=parent_T.plan,
					psi_nmu_parent=parent_T.psi_nmu,
					psi_mun_parent=parent_T.psi_mun,
					band_chunk_size=_chunks_T['band_chunk'],
					q_chunk_size=_chunks_T['q_chunk'],
					bispinor=True,
					band_range_left=band_range_left,
					band_range_right=band_range_right,
					band_norms=_band_norms,
					distributed_cholesky=cfg.backend.distributed_cholesky,
					distributed_lu=cfg.backend.distributed_lu,
					distrib_la_batched_route=_transverse_batched_route,
					zeta_ridge=cfg.backend.zeta_ridge,
					distributed_zeta_solve=cfg.backend.distributed_zeta_solve,
					transverse_zeta_solve=cfg.backend.transverse_zeta_solve,
					transverse_zeta_rcond=cfg.backend.transverse_zeta_rcond,
					gflat_chunk_size=int(_chunks_T.get('gflat_chunk_size', 0)),
					cache_psi_r=bool(_chunks_T.get('cache_psi_r', True)),
					cache_face_y_blocks=bool(
						_chunks_T.get('cache_face_y_blocks', False)),
					vertex_mu_L=mu_L,
					bispinor_lift=(representation.current_lift or "raw"),
					# Transverse ζ IBZ-write activates whenever the
					# bispinor V_q orchestrator iterates IBZ q's — same
					# gate the charge ζ uses,
					# resolved once in the pre-flight so the provenance
					# stamp and this call cannot disagree.
					# Orbit-closure of the transverse centroid set is
					# checked downstream in ``fit_zeta_to_h5``; failure
					# is loud per the bispinor IBZ requirement.
					write_ibz_only=_write_ibz_only_transverse,
					zeta_cutoff_ry=_zeta_cutoff,
					_coupled_mu123_coordinator=coordinator,
					_coupled_rank_gate=(
						(lambda stage: _drain_coupled_rank_findings(
							mu_L, stage))
						if coordinator is not None else None),
					_spill_coupled_gflat_to_host=bool(
						coordinator is not None),
					_stack_coupled_solve_inputs=coordinator is not None,
					print_fn=print_fn,
				)
			_gate_fresh_zeta_rank_findings(
				f"the μ_L={mu_L} transverse ζ fit's rank truncation",
				transverse=True, print_fn=print_fn)
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
			if coordinator is not None:
				coordinator.finish_channel(mu_L)

		with timing.section("gw_jax.zeta_fit_transverse"):
			if _coupled_mu123_enabled:
				print_fn(
					"  [bispinor] coupled μ_L=1,2,3 transverse Zq: one "
					"shared face transform per r chunk; the three solves, write, and "
					"provenance remain ordered μ_L=1→2→3")
				_drop_traced_caches()
				_coordinator = _CoupledMu123ZqCoordinator()
				_errors = {}
				_errors_lock = threading.Lock()

				def _run_coupled_channel(mu_L):
					try:
						_fit_transverse_channel(mu_L, _coordinator)
					except BaseException as exc:
						with _errors_lock:
							_errors.setdefault(mu_L, exc)
						_coordinator.abort(exc)

				_threads = []
				_setup_exc = None
				try:
					# Starting each successor only after its predecessor has
					# reached the r-loop keeps all preparation collectives in the
					# exact accepted μ order on every process.
					for mu_L in (1, 2, 3):
						_thread = threading.Thread(
							target=_run_coupled_channel, args=(mu_L,),
							name=f"lorrax-zq-mu{mu_L}", daemon=False)
						_thread.start()
						_threads.append(_thread)
						_coordinator.wait_channel_prepared(mu_L)
					_coordinator.release_channels()
				except BaseException as exc:
					_setup_exc = exc
					_coordinator.abort(exc)
				finally:
					for _thread in _threads:
						_thread.join()
				if _errors:
					raise _errors[min(_errors)]
				if _setup_exc is not None:
					raise _setup_exc
			else:
				for mu_L in (1, 2, 3):
					if not _reuse_T[mu_L - 1]:
						_drop_traced_caches()
					_fit_transverse_channel(mu_L)

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
	if getattr(meta, 'mu_basis', None) is not None:
		# zeta-file order -> the run's packed centroid order (I/O seam).
		g_head = meta.mu_basis.pack_axis(g_head, 2)
	hc = HeadChannel(g_head=g_head, v_bare=v_bare, v_avg=v_avg,
	                 v_in_V=v_in_V, mult=mult, len2=len2, mode=mode)
	dump = getattr(cfg.head, 'mc_average_placement_vcoul', None)
	if dump:
		hc = head_ratio_from_bgw_dump(dump, hc, bvec=bvec)
	print_fn(hc.summary())
	return hc


def _vcoul_geometry_and_budget(
        cfg, mem_est, print_fn, wfn):
    """Produce the existing Coulomb geometry, cutoff and memory allowance."""
    bvec = CoulombGeometry.from_wfn(wfn).bvec
    if mem_est is None:
        mem_est = {}
    budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory.per_device_gb))
    try:
        from common.gpu_utils import get_device_memory_info
        budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
    except Exception:
        pass
    if cfg.head.bare_coulomb_cutoff is None:
        vcoul_cutoff_ry = float(wfn.ecutwfc)
    else:
        vcoul_cutoff_ry = float(cfg.head.bare_coulomb_cutoff)
    print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry")
    print_fn(f"    V_q budget:    {budget_gb:.2f} GB")
    return bvec, vcoul_cutoff_ry


def _vcoul_transverse_inputs(
        cfg, zeta_h5_path):
    """Produce the authenticated transverse zeta paths for Coulomb projection."""
    zeta_dir = os.path.dirname(zeta_h5_path)
    zeta_T_paths = [
        os.path.join(zeta_dir, f"zeta_q_mu{mu_L}.h5") for mu_L in (1, 2, 3)
    ]
    bispinor_ready = (
        cfg.bispinor and all(os.path.exists(p) for p in zeta_T_paths)
    )
    if cfg.bispinor and not bispinor_ready:
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
    return zeta_dir, zeta_T_paths, bispinor_ready


def _compute_photon_vq(
        bvec, centroid_indices, cfg, mesh_xy, meta, print_fn, sym, vcoul_cutoff_ry, wfn,
        zeta_T_paths, zeta_dir, zeta_h5_path):
    """Produce the four-current Coulomb tiles and their charge and Gamma views."""
    from .v_q_bispinor import (
        tile_dataset_name,
    )
    from file_io.centroids import load_centroids as _load_centroids
    _cents_curr_path = cfg.paths.centroids_file_current
    _, _cent_T_idx_np, _ = _load_centroids(
        _cents_curr_path, meta.fft_grid)
    _cent_T_idx_for_orchestrator = np.asarray(
        _cent_T_idx_np, dtype=np.int32)
    _cent_C_idx_for_orchestrator = (
        np.asarray(jax.device_get(centroid_indices),
                   dtype=np.int32)
        if centroid_indices is not None else None)
    bispinor_h5_path = os.path.join(zeta_dir, "v_q_bispinor.h5")
    print_fn(f"\n  [bispinor] V_q^{{μ_L,ν_L}} → {bispinor_h5_path}")
    with ZetaLoader(zeta_T_paths[0]) as _z_T0:
        n_rmu_T = int(_z_T0.n_rmu_disk)
    n_rmu_C = int(meta.n_rmu)
    with timing.section("gw_jax.V_q_compute"), \
         jax_profile.trace_section("V_q_compute_bispinor"):
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
                _, photon_g0_vectors = compute_V_q_bispinor_g_flat_to_h5(
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
                    sym=sym,
                    centroid_C_idx=_cent_C_idx_for_orchestrator,
                    centroid_T_idx=_cent_T_idx_for_orchestrator,
                    use_ibz=True,
                    tt_head_correction=bool(
                        cfg.head.bispinor_tt_head_correction),
                    bispinor_gw_mode=None,
                    charge_representation=None,
                    spatial_current_representation=None,
                )
    from file_io.tagged_arrays import read_munu_tensor_from_h5
    V_q_raw = read_munu_tensor_from_h5(
        bispinor_h5_path, tile_dataset_name(0, 0), mesh_xy)
    G0_all = photon_g0_vectors[0]
    if not uses_coupled_photon_head(cfg):
        photon_g0_vectors = None
    if int(V_q_raw.shape[-1]) < int(meta.n_rmu_padded):
        pad = int(meta.n_rmu_padded) - int(V_q_raw.shape[-1])
        V_q_raw = jnp.pad(V_q_raw, ((0, 0), (0, pad), (0, pad)))
    if G0_all is not None and int(G0_all.shape[-1]) < int(meta.n_rmu_padded):
        G0_all = jnp.pad(G0_all,
                         ((0, 0), (0, int(meta.n_rmu_padded) - int(G0_all.shape[-1]))))
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
        with ZetaLoader(zeta_h5_path, mesh=mesh_xy) as zeta_io:
            with mesh_xy:
                head_channel = _build_head_channel(
                    zeta_io, cfg=cfg, meta=meta, wfn=wfn, bvec=bvec,
                    mesh_xy=mesh_xy, sym=sym,
                    centroid_indices=_cent_C_idx_for_orchestrator,
                    vcoul_cutoff_ry=vcoul_cutoff_ry,
                    print_fn=print_fn)
    return V_q_raw, G0_all, head_channel, photon_g0_vectors


def _compute_scalar_vq(
        bgw_v_grid_fn, bvec, centroid_indices, cfg, mesh_xy, meta, print_fn, sym,
        vcoul_cutoff_ry, wfn, zeta_h5_path):
    """Produce the scalar Coulomb operator and its live head views."""
    from .compute_vcoul import compute_all_V_q
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
                head_channel = _build_head_channel(
                    zeta_io, cfg=cfg, meta=meta, wfn=wfn, bvec=bvec,
                    mesh_xy=mesh_xy, sym=sym,
                    centroid_indices=_cent_idx_np,
                    vcoul_cutoff_ry=vcoul_cutoff_ry,
                    print_fn=print_fn)
    return V_q_raw, G0_all, head_channel


def _finalize_vq_views(
        G0_all, V_q_raw, head_channel, meta, photon_g0_vectors, print_fn):
    """Produce the packed Coulomb operator and validate its physical invariants."""
    from common.collectives import gather_to_host as _gather_to_host
    G0_gathered = _gather_to_host(G0_all)
    V_qmunu = (meta.mu_basis.pack_operator(V_q_raw)
               if getattr(meta, 'mu_basis', None) is not None else V_q_raw)
    G0 = G0_gathered
    while G0.ndim > 1:
        G0 = G0[0]
    print_fn(f"\n  V_q computed:")
    print_fn(f"    Shape: {V_qmunu.shape}")
    _vq0_trace = float(jnp.trace(V_q_raw[0]).real)
    print_fn(f"    V_q=0 trace: {_vq0_trace:.4f}")
    from common import sanity
    sanity.check_finite("V_q", V_qmunu, print_fn=print_fn)
    sanity.check_positive("V_q[q=0] trace", _vq0_trace, print_fn=print_fn)
    sanity.check_hermitian("V_q[q=0]", V_q_raw[0], print_fn=print_fn)
    sanity.check_q_conjugate_reciprocity(
        "V_q[all q]", V_q_raw, tuple(meta.kgrid), rtol=1e-5,
        print_fn=print_fn)
    sanity.check_finite("V_q G0 (ζ_μ(G=0) at q=0)", G0, print_fn=print_fn)
    return V_qmunu, G0, head_channel, photon_g0_vectors


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None, sym=None, centroid_indices=None):
	"""Produce bare Coulomb and head views; see docs/architecture/four_current_wiring.md."""
	from .compute_vcoul import compute_all_V_q
	photon_g0_vectors = None

	if jax.process_index() == 0:
		os.sync()
	barrier("zeta_flush")
	(bvec, vcoul_cutoff_ry) = _vcoul_geometry_and_budget(
	    cfg, mem_est, print_fn, wfn)
	(zeta_dir, zeta_T_paths, bispinor_ready) = _vcoul_transverse_inputs(
	    cfg, zeta_h5_path)
	if bispinor_ready:
	    (V_q_raw, G0_all, head_channel, photon_g0_vectors) = _compute_photon_vq(
	        bvec, centroid_indices, cfg, mesh_xy, meta, print_fn, sym, vcoul_cutoff_ry, wfn,
	        zeta_T_paths, zeta_dir, zeta_h5_path)
	else:
	    (V_q_raw, G0_all, head_channel) = _compute_scalar_vq(
	        bgw_v_grid_fn, bvec, centroid_indices, cfg, mesh_xy, meta, print_fn, sym,
	        vcoul_cutoff_ry, wfn, zeta_h5_path)
	return _finalize_vq_views(
	    G0_all, V_q_raw, head_channel, meta, photon_g0_vectors, print_fn)




def _prepare_parent_wavefunction_plan(
	cfg, meta, wfn, band_slices, *, sym, centroid_indices, mesh_xy,
	print_fn=print,
):
	"""Require exact typed parent transport for every supported GW consumer."""
	from .centroid_k_unfold import build_centroid_k_unfold_plan

	if (bool(cfg.compute_mode.needs_screening)
			and str(getattr(cfg.screening.diagrams, 'value',
			                cfg.screening.diagrams)) != 'w_rpa'):
		raise ValueError(
			"GATE parent_screening_diagrams: screening_diagrams = "
			f"{getattr(cfg.screening.diagrams, 'value', cfg.screening.diagrams)} "
			"has not been ported to raw parents; use screening_diagrams = w_rpa.")
	plan = build_centroid_k_unfold_plan(
		sym, centroid_indices, meta.fft_grid, mesh_xy,
		nspinor=int(meta.nspinor), parent_k_frac=wfn.kvecs(k=sym.parent_k_domain),
		layout=meta.mu_basis.layout)
	return plan, True, True


def prepare_isdf_and_wavefunctions(
	*, cfg, wfn, sym, meta, centroid_indices, band_slices,
	mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None,
):
	"""ISDF pipeline (non-restart path reads top-to-bottom):

	  1. Resolve the canonical all-channel ζ reuse contract.
	  2. If refitting, ``plan_gflat_chunks`` → band/r/q/G-flat chunk plan.
	  3. ``load_centroids_band_chunked`` → ψ at centroids for [b0, b4).
	  4. ``fit_zeta`` → reuse or write ζ.h5.
	  5. ``compute_V_q`` → V_qmunu, G0 (reads ζ from disk).
	  6. Flush V_q / G0 / enk + W0 placeholder to restart H5 (mode="w").
	  7. Build the downstream Wavefunctions bundle from the same ψ.

	Returns the resident V, wavefunction bundles and any fresh literal-Gamma
	channel vectors needed by the packed photon head.
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

	refuse_unsupported_bispinor_gw(cfg)
	# Same shape, same two-call-site reason: parser-altitude coverage is
	# duplicated here for a hand-built cfg.  No-op at the default (false).
	refuse_unsupported_bispinor_tt_head_correction(cfg)
	from file_io.wfn_basis import WavefunctionBasisReceipt
	representation = resolve_four_current_representation(
		cfg.bispinor, cfg.bispinor_gw)
	# The receipt's band interval is PHYSICAL, not the mesh-padded carrier
	# edge.  ``b_id_4_user`` is the exact loaded WFN boundary; ``b4`` may be
	# rounded past WFN.nbands and names allocation only.
	_basis_band_interval = (
		int(band_slices.b0), int(meta.b_id_4_user))
	charge_basis_receipt = None
	transverse_basis_receipt = None
	basis_wfn_fingerprint_binding = None
	charge_zeta_identity_receipt = None
	photon_g0_vectors = None
	green_parent_carrier = None
	sigma_parent_carrier = None
	basis_T = None

	# THE I/O SEAM.  Files keep the canonical centroid order (grid-agnostic);
	# everything in memory is in the run's packed order (common.centroid_basis).
	# A reader packs, a writer unpacks, nothing in between converts.
	_mu_basis = getattr(meta, 'mu_basis', None)

	def _to_run_order(array, mu_axes, basis=_mu_basis):
		if array is None or basis is None:
			return array
		for axis in mu_axes:
			array = basis.pack_axis(array, axis)
		return array

	def _to_file_order(array, mu_axes, basis=_mu_basis):
		if array is None or basis is None:
			return array
		for axis in mu_axes:
			array = basis.unpack_axis(array, axis)
		return array

	if not cfg.restart:
		# One bounded canonical HDF5 scan supplies every receipt for this
		# loaded source.  In particular a bispinor run must not reopen/sample
		# the WFN independently for its charge and transverse identities.
		from common.parallel_transport import bind_wfn_fingerprint
		basis_wfn_fingerprint_binding = bind_wfn_fingerprint(wfn)
		charge_basis_receipt = WavefunctionBasisReceipt.from_bound_source(
			wfn=wfn,
			wfn_fingerprint_binding=basis_wfn_fingerprint_binding,
			role='charge', bispinor=bool(int(meta.nspinor) == 4),
			bispinor_lift=(representation.charge_lift or "raw"),
			band_interval=_basis_band_interval,
			fft_grid=meta.fft_grid, centroid_fft_idx=centroid_indices,
			n_rmu_logical=int(meta.n_rmu),
			n_rmu_padded=int(meta.n_rmu_padded))
		from common.wfn_transforms import get_enk_bandrange

		with mesh_xy:
			# Resolve the complete charge/bispinor zeta identity before
			# allocating or pricing anything used solely by a fit.  The one
			# contract includes transverse centroid/solver provenance and an
			# independent verdict for every artifact when bispinor=true.
			zeta_contract = _resolve_zeta_fit_contract(
				wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices,
				tmp_dir, print_fn=print0)
			charge_zeta_identity_receipt = charge_zeta_identity(
				zeta_contract.provenance, wfn=wfn,
				wfn_fingerprint_binding=basis_wfn_fingerprint_binding)
			charge_zeta_reused = bool(zeta_contract.reuse_charge)


			_candidate_plan, _, _ = _prepare_parent_wavefunction_plan(
				cfg, meta, wfn, band_slices, sym=sym,
				centroid_indices=centroid_indices, mesh_xy=mesh_xy,
				print_fn=print0)
			_parent_green_plan = _candidate_plan
			chunks = gflat_plan = None
			if not charge_zeta_reused:
				chunks, gflat_plan = _plan_gflat_chunks_for_channel(
					meta=meta, cfg=cfg, band_slices=band_slices,
					mesh_xy=mesh_xy,
					is_bispinor=bool(int(meta.nspinor) == 4),
					n_q_selected=int(np.asarray(sym.q_irr_full_idx).shape[0]),
					parent_route=dict(n_parent=_candidate_plan.n_parent,
					                  parents_only=True), print_fn=print0)
			_parent_zeta_plan = _candidate_plan if chunks is not None else None
			load_band_chunk = (chunks['band_chunk'] if chunks is not None
			                   else zeta_contract.loader_band_chunk)
			with timing.section("gw_jax.load_centroid_wfns"):
				parent_y, parent_x = load_centroids_band_chunked(
					wfn, sym, meta, centroid_indices,
					bool(int(meta.nspinor) == 4), mesh_xy,
					band_range=band_slices.full_range,
					band_chunk_size=load_band_chunk,
					k_chunk_size=(chunks['centroid_k_chunk'] if chunks is not None
					              else zeta_contract.loader_k_chunk),
					bispinor_lift=(representation.charge_lift or "raw"),
					k_domain=sym.parent_k_domain)
			from .wavefunction_bundle import parent_faces
			_parent_green_faces = parent_faces(parent_y, parent_x, mesh_xy=mesh_xy)
			del parent_y, parent_x
			print0("  ψ storage: parents only -- "
			       f"{_candidate_plan.n_parent} raw WFN parents, "
			       f"{_candidate_plan.n_full} full k rows through typed transport.")

			zeta_path, mem_est, transverse_wfn_data = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, band_slices, tmp_dir,
				chunks, print_fn=print0,
				zeta_contract=zeta_contract,
				k_unfold_plan=_parent_zeta_plan,
				psi_nmu_parent=(
					_parent_green_faces[0] if _parent_zeta_plan is not None
					else None),
				psi_mun_parent=(
					_parent_green_faces[1] if _parent_zeta_plan is not None
					else None))
			transverse_basis_receipt = None
			if transverse_wfn_data is not None:
				_meta_receipt_T = transverse_wfn_data['meta']
				transverse_basis_receipt = (
					WavefunctionBasisReceipt.from_bound_source(
						wfn=wfn,
						wfn_fingerprint_binding=(
							basis_wfn_fingerprint_binding),
						role='transverse', bispinor=True,
						bispinor_lift=(representation.current_lift or "raw"),
						band_interval=_basis_band_interval,
						fft_grid=_meta_receipt_T.fft_grid,
						centroid_fft_idx=(
							transverse_wfn_data['centroid_indices']),
						n_rmu_logical=int(_meta_receipt_T.n_rmu),
						n_rmu_padded=int(_meta_receipt_T.n_rmu_padded)))
			# Profiling helper: LORRAX_EXIT_AFTER_ZETA=1 short-circuits
			# the pipeline right after ζ-fit, before the expensive V_q
			# stage.  Combine with LORRAX_MAX_RCHUNKS=N + LORRAX_DEBUG_PRINT=1
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

			wfns_transverse = None
			from .wavefunction_bundle import wavefunctions_face_from_restart
			from common.wfn_transforms import (
				get_enk_bandrange as _get_enk_bandrange_early)
			_enk_full_face, _ = _get_enk_bandrange_early(
				wfn, sym, band_slices.full_range,
				(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)
			with timing.section("gw_jax.wavefunction_setup"):
				wfns = wavefunctions_face_from_restart(
					None, None, enk_full=_enk_full_face,
					slices=band_slices, mesh_xy=mesh_xy,
					basis_receipt=charge_basis_receipt)
			from .wavefunction_bundle import (
				build_packed_parent_green_carrier)
			# The same parent carrier serves screening and band projection.
			_parent_carrier = build_packed_parent_green_carrier(
				wfns, *_parent_green_faces,
				plan=_candidate_plan, mesh_xy=mesh_xy)
			del _parent_green_faces
			sigma_parent_carrier = _parent_carrier
			print0(
				"  Parent-k Sigma route ready: G contracted on "
				f"{_candidate_plan.n_parent} raw WFN parents and Σ_k "
				"projected on their bands, broadcast to "
				f"{_candidate_plan.n_full} full k rows by the typed "
				"band-index unfold.")
			green_parent_carrier = _parent_carrier
			print0(
				"  Parent-k Green contraction ready: "
				f"{_parent_green_plan.n_parent} raw WFN parents -> "
				f"{_parent_green_plan.n_full} full k rows.  Full-k FFTs "
				"and observables remain unchanged.")
			print0(f"  Wavefunctions built: {band_slices.nb_full} bands on raw parents.")
			if transverse_wfn_data is not None:
				parent_T = transverse_wfn_data['green_parent']
				with timing.section("gw_jax.wavefunction_setup"):
					wfns_transverse = wavefunctions_face_from_restart(
						None, None,
						enk_full=_enk_full_face,
						slices=band_slices, mesh_xy=mesh_xy,
						basis_receipt=transverse_basis_receipt)
				wfns_transverse = replace(
					wfns_transverse, green_parent=parent_T)
				if transverse_basis_receipt is not None:
					transverse_basis_receipt.assert_matches_carrier(
						wfns_transverse, where="fresh current parent faces")
				print0(f"  [bispinor] σ^B-side Wfns built on "
				       f"n_rmu_T={transverse_wfn_data['meta'].n_rmu} "
				       f"transverse centroids (face layout; "
				       f"low_mem_bands=true)")

			basis_T = (None if transverse_wfn_data is None
			           else transverse_wfn_data["meta"].mu_basis)

			# P4 — pre-V_q.  Whatever's still in HBM after fit_zeta
			# returns forms the persistent baseline that V_q's transient
			# peak stacks on top of.  Same env gate as the ζ-fit probes
			# (the one driver debug stream).  Round-1 addition.
			from gw.isdf_fitting import mem_probe as _mem_probe
			_mem_probe("pre_v_q")
			V_qmunu, G0, head_channel, photon_g0_vectors = compute_V_q(
				zeta_path, wfn, meta, mesh_xy, cfg,
				mem_est=mem_est, print_fn=print0,
				bgw_v_grid_fn=bgw_v_grid_fn,
				sym=sym, centroid_indices=centroid_indices)
			if photon_g0_vectors is not None:
				photon_g0_vectors = tuple(
					(meta.mu_basis if channel == 0 else basis_T).pack_axis(vector, -1)
					for channel, vector in enumerate(photon_g0_vectors))
			# P5 — post-V_q.  V_q's transient peak just happened inside
			# compute_V_q; this probe captures what survives (V_qmunu,
			# G0) plus anything held over from ζ-fit.  Combined with P4
			# and the V_q HLO buffer-assignment.txt this lets us model
			# V_q's contribution to overall HBM peak.  Round-1 addition.
			if debug_print_enabled():
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
				from file_io.qp_wfn import (
					qp_state_source_provenance_from_binding)
				write_restart_state_to_h5(
					tensors_filename,
					n_rmu_logical=int(meta.n_rmu),
					# THE COULOMB-KERNEL POLICY, stamped with the tensors it
					# describes.  V_qmunu is reused verbatim by every later
					# restart and compute_V_q never re-runs, so without this
					# an averaging-policy change is inherited in silence with
					# every other guard passing.
					coulomb_policy=coulomb_policy_from_config(cfg, meta),
					V_qmunu=_to_file_order(V_qmunu, (-2, -1)),
					G0_mu_nu=G0, enk_full=enk_full,
					init_W0=True, mesh=mesh_xy,
					mode="w", kgrid=tuple(int(v) for v in meta.kgrid),
					qp_state_source_record=(
						qp_state_source_provenance_from_binding(
						wfn,
						wfn_fingerprint_binding=(
							basis_wfn_fingerprint_binding))),
					charge_zeta_identity=charge_zeta_identity_receipt,
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

			# Append the two canonical raw-parent faces of each present family.
			if _write_restart:
				parent_T = None if wfns_transverse is None else wfns_transverse.green_parent
				basis_T = None if transverse_wfn_data is None else transverse_wfn_data['meta'].mu_basis
				write_restart_state_to_h5(
					tensors_filename,
					n_rmu_logical=int(meta.n_rmu),
					psi_parent_y=_to_file_order(sigma_parent_carrier.psi_nmu, (3,)),
					psi_parent_y_mun=_to_file_order(sigma_parent_carrier.psi_mun, (2,)),
					parent_k_rows=_candidate_plan.parent_full_rows,
					mesh=mesh_xy, mode="a",
					psi_parent_y_transverse=(
						_to_file_order(parent_T.psi_nmu, (3,), basis_T) if parent_T is not None else None),
					psi_parent_y_transverse_mun=(
						_to_file_order(parent_T.psi_mun, (2,), basis_T) if parent_T is not None else None),
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
		# low_mem_bands, including bispinor, is supported on this restart
		# branch.  The entry resolver above is a centralized compatibility
		# hook whose current deck-key refusal table is empty.
		from file_io import load_restart_state_from_h5
		from file_io.qp_wfn import (
			authenticate_restart_qp_state_source_for_wfn)
		_restart_source_record, basis_wfn_fingerprint_binding = (
			authenticate_restart_qp_state_source_for_wfn(
			wfn=wfn,
			state_artifact_path=tensors_filename,
			where="gw_jax restart"))
		# The owner above returns the exact record it read and authenticated;
		# do not reopen the artifact across that trust boundary.  Absence remains
		# usable by legacy GW paths, but cannot support a new immutable basis
		# receipt: there is no fact tying the stored psi/E bytes to this WFN.
		_restart_wfn_provenance_complete = (
			_restart_source_record is not None)
		with timing.section(
				"gw_jax.restart_load", announce=True,
				label="restart load (metadata, SlabIO tensors, wedge, reshard)"):
			with h5py.File(tensors_filename, 'r') as header:
				names = ('psi_parent_y', 'psi_parent_y_mun')
				if cfg.bispinor:
					names += ('psi_parent_y_transverse', 'psi_parent_y_transverse_mun')
				if not all(name in header for name in names) or any(
						name in header for name in ('psi_full_y', 'psi_full_y_mun',
						'psi_full_y_transverse', 'psi_full_y_transverse_mun')):
					raise ValueError(
						"GW restart requires raw-parent wavefunctions; this file stores "
						"an incomplete or retired full-k carrier. Rerun with restart = false.")
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices,
				n_rmu_logical=int(meta.n_rmu),
				low_mem_bands=True)
			charge_zeta_identity_receipt = rs.charge_zeta_identity
			# Files keep the canonical centroid order: convert at the seam.
			V_qmunu = _to_run_order(rs.V_qmunu, (-2, -1))
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
			sanity.check_finite("restart ψ (psi_parent_y)",
			                    rs.psi_nmu_parent, print_fn=print0)
			sanity.check_finite("restart ψ (psi_parent_y_mun)",
			                    rs.psi_mun_parent, print_fn=print0)
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
					f"with restart = false.  This legacy bundle carries NO "
					f"WavefunctionBasisReceipt, so authenticated endpoint "
					f"consumers will refuse. ***")
			elif _have_c != _centroid_table_md5(centroid_indices):
				raise ValueError(
					f"restart: {tensors_filename} was written for a "
					f"DIFFERENT charge centroid table (md5 {_have_c} on "
					f"disk vs {_centroid_table_md5(centroid_indices)} "
					f"from this run's centroids_file).  Same count, "
					f"different points ⇒ ψ/V_q sampled at the wrong r_μ "
					f"(silently wrong physics).  Set restart = false, or "
					f"restore the original centroid file.")
			elif not _restart_wfn_provenance_complete:
				print0(
					f"  *** LORRAX SANITY: {tensors_filename} carries a "
					f"matching charge centroid stamp but no canonical "
					f"qp_state_source_provenance record.  Legacy restart "
					f"consumers remain available, but this bundle carries NO "
					f"WavefunctionBasisReceipt because its stored psi/E source "
					f"cannot be authenticated; finite-transfer current "
					f"construction will refuse. ***")
			else:
				charge_basis_receipt = (
					WavefunctionBasisReceipt.from_bound_source(
					wfn=wfn,
					wfn_fingerprint_binding=(
						basis_wfn_fingerprint_binding),
					role='charge',
					bispinor=bool(int(meta.nspinor) == 4),
					band_interval=_basis_band_interval,
					fft_grid=meta.fft_grid,
					centroid_fft_idx=centroid_indices,
					n_rmu_logical=int(meta.n_rmu),
					n_rmu_padded=int(meta.n_rmu_padded)))
			from .wavefunction_bundle import wavefunctions_face_from_restart
			wfns = wavefunctions_face_from_restart(
				None,
				None, enk_full=rs.enk_full,
				slices=band_slices, mesh_xy=mesh_xy,
				basis_receipt=charge_basis_receipt)
			from .wavefunction_bundle import build_packed_parent_green_carrier
			_restart_plan, _, _ = _prepare_parent_wavefunction_plan(
				cfg, meta, wfn, band_slices, sym=sym,
				centroid_indices=centroid_indices, mesh_xy=mesh_xy,
				print_fn=print0)
			if (_restart_plan.parent_full_rows is None
					or not np.array_equal(
						np.asarray(_restart_plan.parent_full_rows),
						np.asarray(rs.parent_k_rows))):
				raise ValueError(
					f"restart: {tensors_filename} names parent rows "
					f"{np.asarray(rs.parent_k_rows).tolist()} but this "
					"WFN's plan names "
					f"{None if _restart_plan.parent_full_rows is None else np.asarray(_restart_plan.parent_full_rows).tolist()}; "
					"the file was written from a different WFN or "
					"symmetry set.")
			sigma_parent_carrier = build_packed_parent_green_carrier(
				wfns, _to_run_order(rs.psi_nmu_parent, (3,)),
				_to_run_order(rs.psi_mun_parent, (2,)),
				plan=_restart_plan, mesh_xy=mesh_xy)
			green_parent_carrier = sigma_parent_carrier
			print0(
				"  ψ storage: parents only (restart) -- "
				f"{_restart_plan.n_parent} raw WFN parents read from "
				f"{os.path.basename(tensors_filename)} stand in for "
				f"{_restart_plan.n_full} full k rows in screening and "
				"Σ through the typed local unfold; no full-k face "
				"was formed.")
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
			# The transverse parent pair is required by the header guard above.
			wfns_transverse = None
			if cfg.bispinor:
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
				transverse_basis_receipt = None
				from common.centroid_basis import PackedCentroidBasis
				basis_T = PackedCentroidBasis.build(
					_cent_T_idx_now, sym, meta.fft_grid, mesh_xy)
				_n_rmu_T_padded = basis_T.n_packed
				meta_restart_transverse = replace(
					meta, n_rmu=int(_n_rmu_curr_now), mu_basis=basis_T,
					n_rmu_padded=_n_rmu_T_padded, nspinor=4, npol=4)
				meta_restart_transverse.sys_dim = meta.sys_dim
				meta_restart_transverse.bispinor = True
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
						f"restart = false.  This legacy transverse bundle "
						f"carries NO WavefunctionBasisReceipt. ***")
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
				elif not _restart_wfn_provenance_complete:
					print0(
						f"  *** LORRAX SANITY: {tensors_filename} carries a "
						f"matching transverse centroid stamp but no canonical "
						f"qp_state_source_provenance record.  The transverse "
						f"bundle carries NO WavefunctionBasisReceipt; "
						f"finite-transfer current construction will refuse. ***")
				else:
					transverse_basis_receipt = (
						WavefunctionBasisReceipt.from_bound_source(
							wfn=wfn,
							wfn_fingerprint_binding=(
								basis_wfn_fingerprint_binding),
							role='transverse', bispinor=True,
							band_interval=_basis_band_interval,
							fft_grid=meta.fft_grid,
							centroid_fft_idx=_cent_T_idx_now,
							n_rmu_logical=int(_n_rmu_curr_now),
							n_rmu_padded=_n_rmu_T_padded))
				from .wavefunction_bundle import (
					wavefunctions_face_from_restart)
				wfns_transverse = wavefunctions_face_from_restart(
					None,
					None,
					enk_full=rs.enk_full, slices=band_slices,
					mesh_xy=mesh_xy, basis_receipt=transverse_basis_receipt)
				plan_T, _, _ = _prepare_parent_wavefunction_plan(
					cfg, meta_restart_transverse, wfn, band_slices, sym=sym,
					centroid_indices=_cent_T_idx_now, mesh_xy=mesh_xy,
					print_fn=print0)
				carrier_T = build_packed_parent_green_carrier(
					wfns_transverse,
					_to_run_order(rs.psi_nmu_parent_transverse, (3,), basis_T),
					_to_run_order(rs.psi_mun_parent_transverse, (2,), basis_T),
					plan=plan_T, mesh_xy=mesh_xy)
				wfns_transverse = replace(wfns_transverse, green_parent=carrier_T)
				if transverse_basis_receipt is not None:
					transverse_basis_receipt.assert_matches_carrier(
						wfns_transverse, where="restart current parent faces")
				print0(f"  [bispinor] σ^B-side Wfns rebuilt from restart "
				       f"(raw parents, "
				       f"n_rmu_T={int(rs.n_rmu_transverse_disk)} "
				       f"transverse centroids)")

		# Recompute the head from canonical one-leg factors, never file padding.
		if uses_coupled_photon_head(cfg):
			from file_io.slab_io import SlabIO
			from jax.sharding import PartitionSpec
			head_path = os.path.join(tmp_dir, "v_q_bispinor.h5")
			with h5py.File(head_path, "r") as header:
				for channel in range(4):
					basis = meta.mu_basis if channel == 0 else basis_T
					name = f"photon_g0_vectors_{channel}"
					if name not in header or header[name].shape != (1, basis.n_logical):
						raise ValueError(
							"Packed photon restart requires canonical Gamma vectors; "
							"rerun restart=false to write the current schema.")
			with SlabIO(head_path, mode="r", mesh=mesh_xy) as head_io:
				photon_g0_vectors = tuple(
					_to_run_order(head_io.read_slab(
						f"photon_g0_vectors_{channel}",
						shape=(1, basis.n_canonical),
						partition_spec=PartitionSpec(None, "x"), dtype=jnp.complex128),
						(1,), basis)
					for channel, basis in enumerate((meta.mu_basis, basis_T, basis_T, basis_T)))

	if green_parent_carrier is not None:
		wfns = replace(wfns, green_parent=green_parent_carrier)
	from .wavefunction_bundle import AuthenticatedWavefunctions
	charge_basis_binding = (
		None if charge_basis_receipt is None else
		AuthenticatedWavefunctions(wfns, charge_basis_receipt))
	transverse_basis_binding = (
		None if transverse_basis_receipt is None else
		AuthenticatedWavefunctions(
			wfns_transverse, transverse_basis_receipt))
	return SimpleNamespace(
		V_qmunu=V_qmunu,
		wf_bundle=wfns,
		wf_bundle_transverse=wfns_transverse,
		mu_bases=(meta.mu_basis, basis_T),
		# Raw-parent, orbit-packed ψ is a screening-only acceleration carrier,
		# deliberately separate from the primary Wavefunctions pytree so head,
		# Sigma, density and output kernels cannot inherit unused large inputs.
		green_parent_carrier=green_parent_carrier,
		sigma_parent_carrier=sigma_parent_carrier,
		n_rmu_charge_logical=int(meta.n_rmu),
		n_rmu_transverse_logical=(
			int(transverse_basis_receipt.n_rmu_logical)
			if transverse_basis_receipt is not None else 0),
		# The exact loaded-WFN identity was already scanned while binding the
		# charge/transverse basis receipts.  Keep that opaque host proof at the
		# orchestration seam so later artifact gates cannot reopen/resample the
		# same WFN.  Legacy restart bundles legitimately carry ``None``.
		wfn_fingerprint_binding=basis_wfn_fingerprint_binding,
		charge_zeta_identity=charge_zeta_identity_receipt,
		# Host-only provenance stays in these orchestration bindings rather
		# than entering Wavefunctions' JAX pytree.  QP rotation returns only a
		# numerical carrier, so it cannot accidentally inherit a DFT binding.
		wf_binding_charge=charge_basis_binding,
		wf_binding_transverse=transverse_basis_binding,
		head_channel=head_channel,
		photon_g0_vectors=photon_g0_vectors,
	)

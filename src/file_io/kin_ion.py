"""Kinetic + ionic Hamiltonian I/O, and the mean-field V_H it may carry.

``kin_ion`` holds ⟨mk|T + V_loc + V_NL|nk⟩ — **pristine ionic mean field,
never V_H** in files written by the current ``gw.kin_ion_io``.

H₀ = kin_ion + V_H is a ~500 eV catastrophic cancellation, so *where*
⟨mk|V_H|nk⟩ comes from is a first-class, explicitly-resolved decision.
Three sources, in ``auto`` precedence order:

``stored``  — the file carries a separate ``v_hartree`` dataset
    ``(nk, nb, nb)`` complex128, Ry, the exact FFT-grid matrix evaluated
    at generation time (this is what ``gw.kin_ion_io`` writes by
    default).  The **full matrix**, not just the diagonal, so a QSGW
    band rotation can transform it.  ``kin_ion`` itself stays pristine.
``isdf``    — ⟨mk|V_H|nk⟩ from the ISDF ``V_q[0]`` tile
    (``gw.cohsex_sigma``'s Hartree kernel).  Fully P-distributed and
    recomputable in-loop; centroid-count dependent.
``gspace``  — built on the fly by the driver through the same exact
    FFT-grid route (``gw.kin_ion_io.compute_hartree_matrix``), on the
    run's own device mesh.  Since scorecard §X that route is
    distributed (ρ: one psum; Poisson: replicated; matrix elements:
    k-partitioned + one gather), so ``gspace`` is now the in-loop /
    QSGW-capable spelling of the exact V_H and not just an offline one.

Plus one **legacy** state that only ever appears on disk, never as a
request:

``folded``  — ``has_hartree=True`` and NO ``v_hartree`` dataset: V_H was
    added *into* the ``kin_ion`` values (the pre-``v_hartree`` format).
    The driver must then add no V_H of its own.  Read-only support.

**Back-compatibility is safe in the dangerous direction.**  A file in
the new format has pristine ``kin_ion`` and no ``has_hartree`` attribute,
so an *old* reader treats it as a legacy ionic-only file and adds its own
ISDF V_H — which is correct, not a double count.  Only the reverse
(a ``folded`` file read as pristine) would double count, and
``_warn_on_unphysical_h0`` fires hard on that.

Reading the attributes / dataset presence is the ONLY supported way to
tell these apart — never infer from magnitudes.

WHICH k-SET THE ARRAYS ARE STORED ON — read this before indexing one
----------------------------------------------------------------------
Both ``kin_ion`` and ``v_hartree`` are computed on the STAR WEDGE — one
row per symmetry orbit — and, since the store-compressed change,
**stored** there too: the file's k axis is ``n_orbits`` rows, not ``nk``,
and the full-BZ table is rebuilt by :func:`broadcast_ibz_to_full_bz` when
the array is read.  The reduction is the star count — 8× on the Si 4³/48-op
decks, 1× on a deck whose every k is its own star — and it is exact by
construction, because what is persisted is the very block the sweep
produced one statement before the broadcast consumed it.

THE STAR WEDGE IS NOT ALWAYS THE WFN'S OWN k-SET, and the stored
``irr_idx_k`` indexes the STORED ROWS, not ``SymMaps.irr_idx_k``'s
upstream wedge labels: ``gnppm_debug`` stores 9 k over 5 orbits, and a
table filed verbatim there would claim 9 stored rows for a 5-row slab.
Both writers renumber through
:func:`file_io.sigma_output.compact_star_tables`, and
:func:`read_star_map` refuses a file where the two disagree.

A file says so in the ``k_storage`` attr of each dataset, and carries the
two tables the rebuild needs (:data:`IRR_IDX_DATASET`,
:data:`SYM_IDX_DATASET`) beside them.  **A dataset with no ``k_storage``
attr is read as ``"full"``**, so every restart file, committed fixture and
hand-written test file that predates the change keeps working untouched —
and, just as important, is never *reinterpreted*: the four older committed
fixtures were computed independently at every full-BZ k and their rows do
NOT satisfy the star relation (measured max|Δ| 3.6e-14 … 7.8e+00 Ry), so
silently treating them as compressible would move physics.  It cannot,
because the discriminator is an attribute the old writer never wrote.
"""
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .slab_io import SlabIO

#: Name of the separate exact-V_H dataset inside ``kin_ion.h5``.
HARTREE_DATASET = "v_hartree"

#: Legal values of the ``hartree_source`` input key.
HARTREE_SOURCES = ("auto", "stored", "isdf", "gspace")

#: Per-dataset attr naming the k-set the array is STORED on.  **Absent
#: means** :data:`K_STORAGE_FULL` — see the module docstring for why that
#: default is load-bearing rather than merely convenient.
K_STORAGE_ATTR = "k_storage"
K_STORAGE_IBZ = "ibz"
K_STORAGE_FULL = "full"
K_STORAGE_VALUES = (K_STORAGE_IBZ, K_STORAGE_FULL)

#: Format version of the IBZ storage.  Stamped on every ``k_storage="ibz"``
#: dataset; a reader refuses a version it was not written against rather
#: than guessing at a table layout.
K_STORAGE_VERSION_ATTR = "k_storage_version"
K_STORAGE_VERSION = 1

#: The two unfold tables, stored in the same file as the arrays they
#: unfold.  A table that lives elsewhere is a table that silently decays
#: when anything upstream is regenerated.
IRR_IDX_DATASET = "irr_idx_k"
SYM_IDX_DATASET = "sym_idx_k"

#: Attr carrying ``n_sym_spatial`` (= ``ntran``): rows of the TRS-augmented
#: symmetry table at or past it are antiunitary, and that is the whole
#: content of the conjugation predicate.
N_SYM_SPATIAL_ATTR = "n_sym_spatial"


# ===========================================================================
# THE UNFOLD, AND WHY THE PREDICATE IS PASSED EXPLICITLY
# ===========================================================================
# ``kin_ion`` (T + V_loc + V_NL) and ``v_hartree`` (V_H) are SCALAR operators
# built from the crystal's own potentials, so each commutes with every
# operation of the space group AND with time reversal, and the full-BZ table
# holds only ``nrk`` distinct matrices.  ``gw.kin_ion_io`` derives that
# statement in full; what matters at the READ side is the shape of the map:
#
#   UNITARY ROW (sym_idx < n_sym_spatial) — a pure copy of the parent's
#     matrix.  The same R acts on bra and ket and O commutes with it, so no
#     rotation, no phase and no degenerate-subspace unitary is left over.
#   ANTIUNITARY ROW (sym_idx >= n_sym_spatial) — the ELEMENT-WISE conjugate
#     of the parent's matrix, because ⟨Θm|O|Θn⟩ = conj⟨m|O|n⟩.
#
# THE PREDICATE, AND WHY IT IS PASSED EXPLICITLY.  The rule is
# ``sym_idx_k >= n_sym_spatial``, i.e. "is this row time-reversed", because
# the reference is the file's own IBZ slab — written verbatim by the
# generator with no symmetry operation applied, so the representative is
# untransformed by construction.  ``star_broadcast``'s DEFAULT predicate is
# a different one (the XOR against the first FULL-BZ row of the star),
# correct for the ``star_select`` output it is normally handed and wrong
# here.  The two stopped coinciding: on ``tests/regression/cohsex_debug``
# the shipping op-selection policy gives star label 2 a first full-BZ row
# with sym_idx 12 = ntran (a pure time reversal), so the XOR inverts the
# conjugation on 6 of the 9 k-points — MEASURED 183.61 eV of error in
# ⟨m|V_H|n⟩, entirely in the OFF-DIAGONALS, with the diagonal and therefore
# every diagonal observable exactly unchanged, which is why nothing caught
# it.  Hence ``trs_reference="ibz_slab"`` is passed explicitly, and
# ``tests/test_kin_ion_star_broadcast.py::
# test_the_call_site_passes_ibz_slab_as_a_literal`` parses THIS FILE to
# check that it still is.
#
# The pin moved here with the function.  It used to parse
# ``src/gw/kin_ion_io.py``, which is where the broadcast lived while it
# happened at WRITE time; the same cell now also asserts that no
# ``star_broadcast`` call is left over there, so the predicate cannot be
# re-introduced at the writer under a different literal.


def broadcast_ibz_to_full_bz(A_irr, irr_idx_k, sym_idx_k, n_sym_spatial):
	"""``(n_orbits, …) → (nk_tot, …)`` through the star map, conj on TRS.

	THE adapter over :func:`symmetry_maps.star_broadcast`, so the
	time-reversal rule has ONE implementation in the tree — one call site,
	reached by both the reader here and ``gw.kin_ion_io``'s writer-side
	wrapper.  ``star_broadcast`` orders ``A_irr`` by ``star_select``'s
	first-occurrence rows; the rows here are the file's own stored rows in
	that order, and ``irr_idx_k`` was renumbered against them by the
	writer, so the labels passed are the identity — which makes its gather
	``A[irr_idx_k]``, with ``conj`` applied on the time-reversed rows.

	``None`` in, ``None`` out: the writer's callers gather with
	``owner_only=True``, so the peers hold no table to broadcast.

	A device operand stays on its device; nothing here pulls the array to
	the host, which is what lets the read path unfold a replicated slab in
	place.
	"""
	if A_irr is None:
		return None
	parent = np.asarray(irr_idx_k, dtype=int)
	n_rows = int(np.shape(A_irr)[0])
	if int(parent.max(initial=-1)) >= n_rows:
		raise ValueError(
			f"broadcast_ibz_to_full_bz: irr_idx_k reaches IBZ row "
			f"{int(parent.max())} but the table has only {n_rows} rows — "
			f"the sweep did not run on the IBZ k-set, or the stored slab "
			f"is truncated against the tables filed with it.")
	# THE MODULE BINDING, not ``from symmetry_maps import star_broadcast``.
	# The AST gate finds this call by ``func.attr == "star_broadcast"``; a
	# bare-name import would make its search find zero calls, and the cell
	# asserts ``len(calls) == 1``, so it FAILS LOUDLY with the count rather
	# than passing on an empty search.  Lazy, as every other lorrax→service
	# edge in this tree is, so importing ``file_io`` costs no service import.
	from ffi import _services
	_services.ensure_on_path()
	import symmetry_maps
	return symmetry_maps.star_broadcast(
		A_irr, parent, np.asarray(sym_idx_k), int(n_sym_spatial),
		irr_labels=np.arange(n_rows, dtype=np.int32),
		trs_reference="ibz_slab")


def read_star_map(h5_path: str, dataset: str = "kin_ion", *, k_axis: int = 0):
	"""The unfold tables a stored-on-IBZ ``dataset`` needs, or ``None``.

	``None`` means the dataset is stored on the full BZ and must be read
	verbatim — which is what a missing :data:`K_STORAGE_ATTR` means, so
	every file written before this format keeps its meaning exactly.

	Otherwise returns ``(irr_idx_k, sym_idx_k, n_sym_spatial)``.  Every
	way the file can be internally inconsistent raises here rather than
	downstream: a claim with no tables, tables of different lengths, a
	version this reader was not written against, a storage value that is
	neither of the two legal ones, and — the one that matters most — a
	stored k axis that does not match the number of distinct stars the
	tables describe, which is what a truncated or mislabelled slab looks
	like from the outside.

	``k_axis`` names WHICH axis of the stored dataset is the k axis, and
	it exists because ``sigma_mnk.h5``'s dynamic cubes are
	``(n_omega, nk, nb, nb)`` — axis 0 there is frequency, and a check
	that read ``shape[0]`` would compare the star count against the
	frequency count and refuse every correctly-written cube.  It defaults
	to 0, which is every ``kin_ion.h5`` array, so no existing caller
	changes.  ONE implementation of the stamp contract for both files was
	the point: a second copy in ``sigma_output`` would be a second place
	for the version, the table names and the refusals to drift.
	"""
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		ds = h5[dataset]
		stored = str(ds.attrs.get(K_STORAGE_ATTR, K_STORAGE_FULL))
		if stored not in K_STORAGE_VALUES:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset}.{K_STORAGE_ATTR} is "
				f"{stored!r}, which is neither {K_STORAGE_IBZ!r} nor "
				f"{K_STORAGE_FULL!r}.  A reader that guessed here would pick "
				f"between reading nrk rows as nk and the reverse.")
		if stored == K_STORAGE_FULL:
			return None
		version = int(ds.attrs.get(K_STORAGE_VERSION_ATTR, -1))
		if version != K_STORAGE_VERSION:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} is stored on the IBZ "
				f"at format version {version}, but this reader implements "
				f"version {K_STORAGE_VERSION}.  Regenerate kin_ion.h5.")
		if N_SYM_SPATIAL_ATTR not in ds.attrs:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} claims IBZ storage "
				f"but carries no {N_SYM_SPATIAL_ATTR!r} attr, so the "
				f"conjugation predicate has no threshold to test against.")
		n_sym_spatial = int(ds.attrs[N_SYM_SPATIAL_ATTR])
		missing = [n for n in (IRR_IDX_DATASET, SYM_IDX_DATASET)
				   if n not in h5]
		if missing:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} claims IBZ storage "
				f"but the file carries no {missing} — the tensor cannot be "
				f"unfolded at all.  A tensor whose reconstruction tables "
				f"live elsewhere is a tensor that silently decays.")
		irr = np.asarray(h5[IRR_IDX_DATASET][()], dtype=np.int32)
		sidx = np.asarray(h5[SYM_IDX_DATASET][()], dtype=np.int32)
		if not (-ds.ndim <= k_axis < ds.ndim):
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} has {ds.ndim} axes, "
				f"so k_axis={k_axis} names no axis of it.")
		nk_stored = int(ds.shape[k_axis])
	if irr.shape != sidx.shape or irr.ndim != 1:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {IRR_IDX_DATASET} {irr.shape} and "
			f"{SYM_IDX_DATASET} {sidx.shape} must both be (nk_full,)")
	# THE NUMBER OF DISTINCT STARS, not ``max + 1``.  The two agree only
	# while the labels are dense, which is exactly what the writers are
	# supposed to guarantee (``file_io.sigma_output.compact_star_tables``,
	# and ``gw.kin_ion_io.star_tables`` through it) — so testing ``max+1``
	# tested the writers' arithmetic instead of their output, and passed
	# on the one shape this refusal exists to catch.  MEASURED 2026-08-17:
	# ``gnppm_debug``'s ``irr_idx_k = [0,2,2,6,8,7,6,7,8]`` gives
	# ``max+1 = 9`` against 9 stored rows and sails through, while the
	# true star count is 5 and four of those rows were in a basis nothing
	# reads.  ``np.unique`` costs a sort of ``nk_full`` int32 once per
	# open and is the property the docstring already claimed.
	n_star = int(np.unique(irr).size)
	if n_star != nk_stored:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {dataset} stores {nk_stored} k "
			f"rows but {IRR_IDX_DATASET} describes {n_star} stars over "
			f"{irr.size} full-BZ k.  The slab and the tables filed with it "
			f"do not describe the same calculation — refusing rather than "
			f"unfolding {min(n_star, nk_stored)} of them.")
	if int(sidx.max(initial=-1)) >= 2 * n_sym_spatial:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {SYM_IDX_DATASET} reaches "
			f"{int(sidx.max())} but the table is 2·{N_SYM_SPATIAL_ATTR} = "
			f"{2 * n_sym_spatial} rows long.")
	return irr, sidx, n_sym_spatial


def _unfold_if_ibz(arr, star):
	"""Apply the star broadcast iff ``star`` says the slab is an IBZ one."""
	if star is None:
		return arr
	return broadcast_ibz_to_full_bz(arr, *star)


def read_full_bz_dataset(h5_path: str, dataset: str = "kin_ion"):
	"""One dataset of ``kin_ion.h5``, on the FULL BZ, as a host array.

	The serial-h5py twin of :func:`load_kin_ion_submatrix`, for the
	host-side consumers that read this file with no device mesh to be
	collective over — ``gw.eqp_bgw`` rebuilds eqp{0,1} straight from files
	and has none.  Whole dataset, no band window: those callers slice it
	themselves.

	It exists so "read kin_ion.h5" has ONE meaning in the tree.  The
	alternative is each host-side reader noticing :data:`K_STORAGE_ATTR`
	for itself, and a reader that did not notice would index an
	``(nrk, nb, nb)`` array with full-BZ k — returning another star's
	matrix on every k past the wedge, or an ``IndexError`` on a deck lucky
	enough to be caught.
	"""
	star = read_star_map(h5_path, dataset)
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		arr = np.asarray(h5[dataset][()])
	return np.asarray(_unfold_if_ibz(arr, star))


def read_kin_ion_provenance(h5_path: str) -> dict:
	"""Return the ``kin_ion`` dataset attributes as a plain dict.

	Missing file or dataset raises; a legacy file simply has fewer keys.
	Values are converted to plain Python/NumPy scalars so callers can
	compare and print them without h5py types leaking out.

	``_shape`` is the STORED shape, so its k extent is ``nrk`` on an
	IBZ-stored file.  ``_nk_logical`` is the k count a consumer will
	actually receive — the two are equal on a full-BZ file and differ by
	the star reduction otherwise, and every check about "does this file
	match my run" wants the logical one.
	"""
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	star = read_star_map(h5_path, "kin_ion")
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		ds = h5["kin_ion"]
		out = {k: v for k, v in ds.attrs.items()}
		out["_shape"] = tuple(int(s) for s in ds.shape)
		out["_k_storage"] = (K_STORAGE_FULL if star is None
		                     else K_STORAGE_IBZ)
		out["_nk_logical"] = (int(ds.shape[0]) if star is None
		                      else int(star[0].size))
		out["_has_v_hartree"] = HARTREE_DATASET in h5
		if out["_has_v_hartree"]:
			out["_hartree_shape"] = tuple(
				int(s) for s in h5[HARTREE_DATASET].shape)
			# The array's OWN attrs win.  ``kin_ion`` also stamps
			# ``hartree_truncation_2d`` (False whenever V_H is not folded
			# into its values), so a ``setdefault`` here would mask the
			# stored array's real Coulomb convention behind that False and
			# make the load-time report say "truncation_2d=False" for a
			# correctly 2D-truncated V_H.
			for k, v in h5[HARTREE_DATASET].attrs.items():
				out[f"hartree_{k}"] = v
	return out


def kin_ion_has_hartree(h5_path: str) -> bool:
	"""True iff V_H is **folded into the ``kin_ion`` values themselves**.

	The legacy (pre-``v_hartree``) contract flag for the
	no-double-counting rule.  It is deliberately NOT true for the current
	format, where V_H lives in its own dataset and ``kin_ion`` is
	pristine — see :func:`kin_ion_hartree_source`, which is what new code
	should call.  Legacy files written before the fold-in carry no
	attribute at all and answer False, preserving their semantics.
	"""
	try:
		attrs = read_kin_ion_provenance(h5_path)
	except (FileNotFoundError, KeyError):
		return False
	if attrs.get("_has_v_hartree"):
		return False           # separate array ⇒ kin_ion values are pristine
	return bool(attrs.get("has_hartree", False))


def kin_ion_hartree_source(h5_path: str) -> str:
	"""What the FILE offers: ``'stored'`` | ``'folded'`` | ``'none'``.

	Pure inspection — no policy.  :func:`resolve_hartree_source` applies
	the request and the precedence.
	"""
	try:
		attrs = read_kin_ion_provenance(h5_path)
	except (FileNotFoundError, KeyError):
		return "none"
	if attrs.get("_has_v_hartree"):
		return "stored"
	if bool(attrs.get("has_hartree", False)):
		return "folded"
	return "none"


def resolve_hartree_source(h5_path: str, requested: str = "auto",
                           *, print_fn=print) -> str:
	"""Decide where ⟨mk|V_H|nk⟩ comes from.  Returns the resolved source.

	``auto`` precedence: **stored** array → legacy **folded** values →
	**isdf**.  An explicit request is honoured, except that it may not
	silently contradict a folded file: a ``folded`` ``kin_ion.h5`` has V_H
	inside its values and *any* other source would double count, so that
	combination raises rather than producing a plausible wrong number.

	Returns one of ``'stored' | 'folded' | 'isdf' | 'gspace'``.  Only
	``'folded'`` means "add nothing"; the other three all supply a V_H
	that the driver adds to a pristine ``kin_ion``.
	"""
	requested = str(requested or "auto").strip().lower()
	if requested not in HARTREE_SOURCES:
		raise ValueError(
			f"hartree_source={requested!r} is not one of {HARTREE_SOURCES}")
	available = kin_ion_hartree_source(h5_path)

	if available == "folded":
		if requested in ("auto", "stored"):
			return "folded"
		raise ValueError(
			f"hartree_source={requested!r} was requested but "
			f"{os.path.basename(h5_path)} is a LEGACY folded file: V_H is "
			"already inside its kin_ion values, so adding a second source "
			"would double count ~500 eV.  Regenerate kin_ion.h5 (the current "
			"writer stores V_H as a separate 'v_hartree' array and leaves "
			"kin_ion pristine), or drop the override.")

	if requested == "auto":
		return "stored" if available == "stored" else "isdf"
	if requested == "stored":
		if available != "stored":
			raise ValueError(
				f"hartree_source=stored but {os.path.basename(h5_path)} has no "
				f"'{HARTREE_DATASET}' dataset.  Regenerate it with "
				"`python -m gw.kin_ion_io` (which stores V_H by default), or "
				"use hartree_source=isdf / gspace.")
		return "stored"
	return requested                      # 'isdf' or 'gspace'


def load_hartree_submatrix(
	h5_path: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None = None,
) -> jax.Array:
	"""Read the stored exact ⟨mk|V_H|nk⟩ sub-window, replicated (Ry).

	Same shape/sharding contract as :func:`load_kin_ion_submatrix` — the
	two are added together to form H₀, so they must come back in the same
	layout, which includes coming back on the same k-set.  Both therefore
	unfold a stored IBZ slab here, at the read boundary.  Raises if the
	dataset is absent; call :func:`resolve_hartree_source` first.
	"""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	star = read_star_map(h5_path, HARTREE_DATASET)
	with h5py.File(h5_path, "r") as h5:
		if HARTREE_DATASET not in h5:
			raise KeyError(
				f"Dataset '{HARTREE_DATASET}' missing from {h5_path}")
		nk_stored, nb_total, nb_total2 = h5[HARTREE_DATASET].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"{HARTREE_DATASET} must be square in band axes; "
			f"got {(nk_stored, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but "
			f"{HARTREE_DATASET} only has {nb_total}.  Regenerate kin_ion.h5 "
			f"with at least -n {band_stop}.")
	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh) as io:
		arr = io.read_slab(
			HARTREE_DATASET,
			shape=(nk_stored, nb, nb),
			offset=(0, band_start, band_start),
			dtype=jnp.complex128,
			mesh=mesh,
			partition_spec=P(None, None, None),
		)
	return _unfold_if_ibz(arr, star)


def validate_kin_ion_against_run(
	h5_path: str,
	*,
	sys_dim: int | None = None,
	nk: int | None = None,
	band_stop: int | None = None,
	nspinor: int | None = None,
	print_fn=print,
) -> dict:
	"""Refuse a ``kin_ion.h5`` that disagrees with the run it feeds.

	``kin_ion`` fixes the Coulomb truncation convention and the band
	window for the whole mean-field side of H₀.  A file generated under
	``sys_dim=3`` silently consumed by a ``sys_dim=2`` run puts a large
	*systematic* error into a ~500 eV cancellation — indistinguishable,
	from the outside, from a basis-convergence problem.  Check it once,
	loudly, at load time.  Legacy files (no provenance attrs) are
	accepted with a note; only explicit disagreements raise.
	"""
	attrs = read_kin_ion_provenance(h5_path)
	stored_sys_dim = attrs.get("sys_dim")
	if sys_dim is not None and stored_sys_dim is not None and (
		int(stored_sys_dim) != int(sys_dim)
	):
		raise ValueError(
			f"kin_ion.h5 was generated with sys_dim={int(stored_sys_dim)} but this "
			f"run uses sys_dim={int(sys_dim)}.  The Coulomb truncation convention "
			"must match — regenerate kin_ion.h5 with the run's own input file."
		)
	# Same legacy contract as sys_dim: the producer stamps ``nspinor``
	# (gw.kin_ion_io), an older file simply lacks the attr and is accepted.
	# A spin-structure mismatch is not a band-count problem the checks
	# below could catch — an nspinor=2 file has ψ†ψ over two components
	# and (with SOC) j-resolved V_NL, so its T+V_ion against a scalar
	# WFN's basis is silently, systematically wrong.
	stored_nspinor = attrs.get("nspinor")
	if nspinor is not None and stored_nspinor is not None and (
		int(stored_nspinor) != int(nspinor)
	):
		raise ValueError(
			f"kin_ion.h5 was generated from an nspinor={int(stored_nspinor)} "
			f"WFN but this run's WFN has nspinor={int(nspinor)}.  T+V_ion "
			"matrix elements are per spin structure — regenerate kin_ion.h5 "
			"from the run's own WFN."
		)
	# The LOGICAL k count, not the stored one: an IBZ-stored file holds nrk
	# rows and hands the run nk of them, and comparing the stored extent
	# would refuse every compressed file on a deck with any symmetry at all.
	if nk is not None and int(attrs["_nk_logical"]) != int(nk):
		raise ValueError(
			f"kin_ion.h5 has nk={int(attrs['_nk_logical'])} "
			f"(stored on {attrs['_k_storage']}, {attrs['_shape'][0]} rows on "
			f"disk) but the run has nk={int(nk)}."
		)
	if band_stop is not None and int(attrs["_shape"][1]) < int(band_stop):
		raise ValueError(
			f"kin_ion.h5 has {int(attrs['_shape'][1])} bands but the run's sigma "
			f"window needs {int(band_stop)}."
		)
	if attrs.get("_has_v_hartree"):
		hs = attrs.get("_hartree_shape", ("?", "?", "?"))
		if band_stop is not None and int(hs[1]) < int(band_stop):
			raise ValueError(
				f"kin_ion.h5 stores V_H for {int(hs[1])} bands but the run's "
				f"sigma window needs {int(band_stop)}.  The two arrays must "
				"cover the same window — regenerate with a larger -n.")
		print_fn(
			f"  kin_ion: pristine T+V_loc+V_NL, plus a stored exact V_H array "
			f"{tuple(hs)} (truncation_2d="
			f"{bool(attrs.get('hartree_truncation_2d', False))})."
		)
	elif bool(attrs.get("has_hartree", False)):
		print_fn(
			"  kin_ion: LEGACY folded file — exact FFT-grid V_H is inside the "
			f"kin_ion values (truncation_2d="
			f"{bool(attrs.get('hartree_truncation_2d', False))}); no V_H will "
			"be added on top."
		)
	else:
		print_fn(
			"  kin_ion: ionic only (no stored V_H) — V_H comes from the ISDF "
			"centroid quadrature, so H0 depends on the centroid count."
		)
	return attrs


def load_kin_ion_submatrix(
	h5_path: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None = None,
) -> jax.Array:
	"""Read the [band_start, band_stop) sub-window of ``kin_ion`` replicated.

	Stored dataset is ``H_DFT − V_xc`` (kinetic + ionic; Hartree only if it
	was added at generation time), so V_xc should not be subtracted again
	downstream.

	The full kin_ion sub-block ``(nk, nb, nb)`` fits comfortably on a single
	device — it is loaded **fully replicated** on ``mesh`` so the
	post-self-energy plumbing can operate on replicated arrays uniformly.
	Goes through :class:`SlabIO` for backend parity with the rest of the
	GW input stack (``zeta_q.h5``, ``sigma_omega.h5``).

	THE STAR BROADCAST HAPPENS HERE.  What is on disk is the ``nrk``-row
	block the generator's sweep produced; the ``(nk, nb, nb)`` this returns
	is its unfold, so every caller sees the k-set it always saw and the
	saving is disk and write time rather than a new contract.  The read is
	the stored extent, so the transport moves ``nrk/nk`` of the bytes it
	used to; the broadcast is one gather on the device the slab landed on.

	Parameters
	----------
	h5_path
		Path to ``kin_ion.h5``.
	band_start, band_stop
		0-based half-open band window; ``band_stop > band_start`` and
		``band_stop ≤ nb_total``.
	mesh
		Device mesh.  Required; every slab read is collective over it.
		allgather backend tolerates ``None`` and returns a host-backed
		replicated JAX array.
	backend
		the allgather backend.

	Returns
	-------
	jax.Array, shape ``(nk, nb, nb)``, dtype ``complex128``, replicated.
	"""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")

	# The tables first: a file that claims IBZ storage and cannot back the
	# claim must refuse BEFORE a collective read is issued against it.
	star = read_star_map(h5_path, "kin_ion")

	# Peek shape so we can validate the slice and pass an explicit
	# (nk, nb, nb) request to read_slab — the FFI backend requires it.
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		nk_stored, nb_total, nb_total2 = h5["kin_ion"].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"kin_ion must be square in band axes; "
			f"got {(nk_stored, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but kin_ion "
			f"only has {nb_total}"
		)

	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh) as io:
		arr = io.read_slab(
			"kin_ion",
			shape=(nk_stored, nb, nb),
			offset=(0, band_start, band_start),
			dtype=jnp.complex128,
			mesh=mesh,
			partition_spec=P(None, None, None),
		)
	return _unfold_if_ibz(arr, star)

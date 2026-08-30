"""Kinetic + ionic Hamiltonian I/O.

``kin_ion`` holds ⟨mk|T + V_loc + V_NL|nk⟩ and must be pristine.  Hartree is
always rebuilt live in G-space by the GW driver.  The reader refuses the old
``has_hartree=True`` folded format because adding the live field to it would
double count a roughly 500 eV cancellation.

WHICH k-SET THE ARRAYS ARE STORED ON — read this before indexing one
----------------------------------------------------------------------
The matrix dataset is computed on the STAR WEDGE — one
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

from common.four_current_model import (
	is_pauli_reference_model,
	is_isometric_kinetic_balance_model,
	resolve_four_current_representation,
)

from .slab_io import SlabIO

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
# ``kin_ion`` (T + V_loc + V_NL) is a SCALAR operator built from the
# crystal's own potentials, so it commutes with every
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
	return out


def _load_matrix_submatrix(
	h5_path: str,
	dataset: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None,
) -> jax.Array:
	"""Format-owned collective slab read plus authenticated star unfold."""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")

	# Authenticate the dataset's own k-storage stamp and star tables before
	# issuing any collective payload read.
	star = read_star_map(h5_path, dataset)
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		nk_stored, nb_total, nb_total2 = h5[dataset].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"{dataset} must be square in band axes; "
			f"got {(nk_stored, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but {dataset} only "
			f"has {nb_total}. Regenerate kin_ion.h5 with at least -n "
			f"{band_stop}.")
	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh) as io:
		arr = io.read_slab(
			dataset,
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
	expected_bispinor: bool,
	expected_bispinor_gw_mode: str | None = None,
	sys_dim: int | None = None,
	nk: int | None = None,
	band_stop: int | None = None,
	nspinor: int | None = None,
	print_fn=print,
) -> dict:
	"""Validate pristine kinetic+ionic provenance before its slab is read."""
	attrs = read_kin_ion_provenance(h5_path)
	if bool(attrs.get("has_hartree", False)):
		raise ValueError(
			"kin_ion.h5 has has_hartree=True: this retired format folds V_H "
			"into kin_ion and would double count the mandatory live G-space "
			"Hartree field. Regenerate a pristine kin_ion.h5.")

	stored_bispinor = attrs.get("bispinor")
	if stored_bispinor is None:
		raise ValueError(
			"kin_ion.h5 has no bispinor provenance; regenerate it from the "
			"run's input deck.")
	if bool(stored_bispinor) != bool(expected_bispinor):
		raise ValueError(
			f"kin_ion.h5 has bispinor={bool(stored_bispinor)} but this run "
			f"uses bispinor={bool(expected_bispinor)}; regenerate it.")

	representation = resolve_four_current_representation(
		expected_bispinor, expected_bispinor_gw_mode)
	if expected_bispinor:
		for attr, expected in (
			("charge_representation", representation.charge_representation),
			("spatial_current_representation",
			 representation.spatial_current_representation),
		):
			stored = attrs.get(attr)
			if stored is not None and str(stored) != str(expected):
				raise ValueError(
					f"kin_ion.h5 {attr}={stored!r}, expected {expected!r}.")
		if (is_pauli_reference_model(expected_bispinor_gw_mode)
				or is_isometric_kinetic_balance_model(
					expected_bispinor_gw_mode)):
			expected_mode = str(getattr(
				expected_bispinor_gw_mode, "value",
				expected_bispinor_gw_mode))
			if str(attrs.get("bispinor_gw_mode")) != expected_mode:
				raise ValueError(
					f"kin_ion.h5 bispinor_gw_mode="
					f"{attrs.get('bispinor_gw_mode')!r}, expected "
					f"{expected_mode!r}.")

	stored_sys_dim = attrs.get("sys_dim")
	if (sys_dim is not None and stored_sys_dim is not None
			and int(stored_sys_dim) != int(sys_dim)):
		raise ValueError(
			f"kin_ion.h5 has sys_dim={int(stored_sys_dim)} but this run "
			f"uses sys_dim={int(sys_dim)}.")
	stored_nspinor = attrs.get("nspinor")
	if (nspinor is not None and stored_nspinor is not None
			and int(stored_nspinor) != int(nspinor)):
		raise ValueError(
			f"kin_ion.h5 has nspinor={int(stored_nspinor)} but this run "
			f"uses nspinor={int(nspinor)}; regenerate it from this WFN.")
	if nk is not None and int(attrs["_nk_logical"]) != int(nk):
		raise ValueError(
			f"kin_ion.h5 has nk={int(attrs['_nk_logical'])} but the run has "
			f"nk={int(nk)}.")
	if band_stop is not None and int(attrs["_shape"][1]) < int(band_stop):
		raise ValueError(
			f"kin_ion.h5 has {int(attrs['_shape'][1])} bands but the run "
			f"needs {int(band_stop)}.")
	print_fn("  kin_ion: pristine T+V_loc+V_NL; Hartree is built live in G-space.")
	return attrs


def load_kin_ion_submatrix(
	h5_path: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None = None,
) -> jax.Array:
	"""Read the [band_start, band_stop) sub-window of ``kin_ion`` replicated.

	Stored dataset is the pristine ``T + V_loc + V_NL`` operator.  Folded
	Hartree files are rejected by :func:`validate_kin_ion_against_run`.

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
	return _load_matrix_submatrix(
		h5_path, "kin_ion", band_start, band_stop, mesh=mesh)

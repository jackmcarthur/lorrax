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
Both writers renumber through ``symmetry_maps.KStarMap.take``, and
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

THE STORED ROW ORDER IS NOT THE WFN's k ORDER.  The star wedge keeps its
stars in first-full-BZ-occurrence order (``KStarMap.rows``), which on a
symmetry-reduced cell is a permutation of the WFN's IBZ list (bulk CrCl3
4³: stored row 6 is WFN k 13, rows 7-13 are WFN k 6-12).  So every
star-wedge file also carries its k lists in crystal coordinates
(:data:`KPOINTS_STORED_DATASET`, the parent k of each stored row;
:data:`KPOINTS_FULL_DATASET`, the full-BZ rows the tables unfold onto),
and the run's readers check the second against their own k list
(``restart_bundle.check_kpoints_against_run``).  A consumer that pairs
stored rows with WFN-ordered data pairs them by the first.
"""
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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

#: The k lists of a star-wedge file, crystal coordinates: ``(n_rows, 3)``,
#: the parent k of each stored row (the WFN k of its star; the full-BZ k
#: under a trivial symmetry view), and ``(nk_full, 3)``, the full-BZ k the
#: tables unfold onto, in the unfolded array's row order.  Absent on files
#: written before 2026-11-16, which stay readable unchecked.
KPOINTS_STORED_DATASET = "kpoints_stored_crys"
KPOINTS_FULL_DATASET = "kpoints_full_crys"


def star_wedge_kpoints(sym):
	"""``(stored, full)`` crystal k lists of a star-wedge slab of this run.

	``stored`` is ``(n_rows, 3)``: the parent k (``unfolded_kpts[kirr_fullids]``,
	the WFN's own k modulo a reciprocal lattice vector) of each star, in
	the stored row order of
	``symmetry_maps.star_wedge_rows``.  ``full`` is ``(nk_tot, 3)``:
	``sym.unfolded_kpts``, the order the unfold returns.
	"""
	from ffi import _services
	_services.ensure_on_path()
	import symmetry_maps
	labels, _ = symmetry_maps.star_wedge_rows(sym)
	full = np.asarray(sym.unfolded_kpts, dtype=np.float64)
	parents = full[np.asarray(sym.kirr_fullids, dtype=np.int64)]
	return parents[np.asarray(labels, dtype=np.int64)], full


def queue_star_kpoints(io, kpoints, *, n_rows, nk_full, where):
	"""Queue the two k lists of a star-wedge file on an open ``SlabIO``.

	``kpoints`` is :func:`star_wedge_kpoints`' pair; the shapes are checked
	against the slab's stored rows and the unfold table's full-BZ extent.
	"""
	if kpoints is None:
		raise ValueError(
			f"{where}: star tables given without their k lists; pass "
			f"kpoints=file_io.kin_ion.star_wedge_kpoints(sym).")
	stored, full = (np.asarray(a, dtype=np.float64) for a in kpoints)
	if stored.shape != (int(n_rows), 3) or full.shape != (int(nk_full), 3):
		raise ValueError(
			f"{where}: k lists {stored.shape} / {full.shape} do not match the "
			f"{int(n_rows)} stored rows and {int(nk_full)} full-BZ rows of "
			f"the star tables; pass file_io.kin_ion.star_wedge_kpoints(sym) "
			f"of the same SymMaps as the tables.")
	io.write_attr(KPOINTS_STORED_DATASET, stored)
	io.write_attr(KPOINTS_FULL_DATASET, full)


# THE UNFOLD and its antiunitary predicate: docs/architecture/
# symmetry_register.md §8, "Operator slabs on the star wedge".  The stored
# slab is the untransformed WFN state of each star, so the predicate is the
# member's own flag (``trs_reference="ibz_slab"``); the star-row XOR rule is
# for ``star_select`` output and was measured 183.61 eV wrong here
# (``cohsex_debug``, off-diagonal ⟨m|V_H|n⟩ only).


def broadcast_ibz_to_full_bz(A_irr, irr_idx_k, sym_idx_k, n_sym_spatial):
	"""``(n_orbits, …) → (nk_tot, …)`` through the star map, conj on TRS.

	THE adapter over :func:`symmetry_maps.star_broadcast`, so the
	time-reversal rule has ONE implementation in the tree — one call site,
	reached by the reader here and by :func:`broadcast_star_wedge`.
	``star_broadcast`` orders ``A_irr`` by ``star_select``'s
	first-occurrence rows; the rows here are the file's own stored rows in
	that order, and ``irr_idx_k`` was renumbered against them by the
	writer, so the labels passed are the identity — which makes its gather
	``A[irr_idx_k]``, with ``conj`` applied on the time-reversed rows.

	``None`` in, ``None`` out.

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




def broadcast_star_wedge(A_irr, sym):
	"""A star-wedge slab ``(n_orbits, …)`` of a live run → ``(nk_tot, …)``.

	The writer-side spelling of :func:`broadcast_ibz_to_full_bz`: the tables
	come from the run's ``SymMaps`` (``symmetry_maps.star_wedge_tables``)
	instead of a file.  ``None`` in, ``None`` out.  A device operand keeps
	its trailing-axis sharding.
	"""
	if A_irr is None:
		return None
	from ffi import _services
	_services.ensure_on_path()
	import symmetry_maps
	return broadcast_ibz_to_full_bz(A_irr, *symmetry_maps.star_wedge_tables(sym))


def write_kin_ion(path, H_irr, *, mesh, nb, star, kpoints, attrs) -> None:
	"""Write ``kin_ion.h5`` through SlabIO from the sweep's shards.

	COLLECTIVE over ``mesh``.  ``H_irr`` is the star-wedge slab
	``(n_orbits, nb_pad, nb_pad)`` with its band axes on the mesh (or a
	replicated host array); SlabIO drops the pad rows past the logical
	``nb``.  ``star`` is ``(irr_idx_k, sym_idx_k, n_sym_spatial)`` already
	renumbered onto the stored rows (``symmetry_maps.star_wedge_tables``); the
	two index tables are filed beside the slab they unfold, and the slab is
	stamped ``k_storage = "ibz"``.  ``kpoints`` is
	:func:`star_wedge_kpoints` of the same ``SymMaps``: the stored and
	full-BZ k lists filed beside the tables.  ``attrs`` are the dataset's
	provenance attributes; every rank passes them and rank 0's copy lands.
	"""
	irr_idx_k, sym_idx_k, n_sym_spatial = star
	n_rows = int(H_irr.shape[0])
	with SlabIO(str(path), mode="w", mesh=mesh) as io:
		io.write_attr(IRR_IDX_DATASET, np.asarray(irr_idx_k, dtype=np.int32))
		io.write_attr(SYM_IDX_DATASET, np.asarray(sym_idx_k, dtype=np.int32))
		queue_star_kpoints(io, kpoints, n_rows=n_rows,
		                   nk_full=np.asarray(irr_idx_k).size, where="write_kin_ion")
		io.create_dataset(
			"kin_ion", shape=(n_rows, int(nb), int(nb)), dtype=np.complex128,
			attrs={K_STORAGE_ATTR: K_STORAGE_IBZ,
			       K_STORAGE_VERSION_ATTR: K_STORAGE_VERSION,
			       N_SYM_SPATIAL_ATTR: int(n_sym_spatial), **dict(attrs)})
		io.write_slab("kin_ion", H_irr)


def _unfold_if_ibz(arr, star):
	"""Apply the star broadcast iff ``star`` says the slab is an IBZ one."""
	if star is None:
		return arr
	return broadcast_ibz_to_full_bz(arr, *star)

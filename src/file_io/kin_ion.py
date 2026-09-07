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

from common.four_current_model import resolve_four_current_representation

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




def _unfold_if_ibz(arr, star):
	"""Apply the star broadcast iff ``star`` says the slab is an IBZ one."""
	if star is None:
		return arr
	return broadcast_ibz_to_full_bz(arr, *star)

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


def read_kin_ion_provenance(h5_path: str) -> dict:
	"""Return the ``kin_ion`` dataset attributes as a plain dict.

	Missing file or dataset raises; a legacy file simply has fewer keys.
	Values are converted to plain Python/NumPy scalars so callers can
	compare and print them without h5py types leaking out.
	"""
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		ds = h5["kin_ion"]
		out = {k: v for k, v in ds.attrs.items()}
		out["_shape"] = tuple(int(s) for s in ds.shape)
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
	backend=None,
) -> jax.Array:
	"""Read the stored exact ⟨mk|V_H|nk⟩ sub-window, replicated (Ry).

	Same shape/sharding contract as :func:`load_kin_ion_submatrix` — the
	two are added together to form H₀, so they must come back in the same
	layout.  Raises if the dataset is absent; call
	:func:`resolve_hartree_source` first.
	"""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	with h5py.File(h5_path, "r") as h5:
		if HARTREE_DATASET not in h5:
			raise KeyError(
				f"Dataset '{HARTREE_DATASET}' missing from {h5_path}")
		nk, nb_total, nb_total2 = h5[HARTREE_DATASET].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"{HARTREE_DATASET} must be square in band axes; "
			f"got {(nk, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but "
			f"{HARTREE_DATASET} only has {nb_total}.  Regenerate kin_ion.h5 "
			f"with at least -n {band_stop}.")
	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh, backend=backend) as io:
		return io.read_slab(
			HARTREE_DATASET,
			shape=(nk, nb, nb),
			offset=(0, band_start, band_start),
			dtype=jnp.complex128,
			mesh=mesh,
			partition_spec=P(None, None, None),
		)


def validate_kin_ion_against_run(
	h5_path: str,
	*,
	sys_dim: int | None = None,
	nk: int | None = None,
	band_stop: int | None = None,
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
	if nk is not None and int(attrs["_shape"][0]) != int(nk):
		raise ValueError(
			f"kin_ion.h5 has nk={int(attrs['_shape'][0])} but the run has nk={int(nk)}."
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
	backend=None,
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

	Parameters
	----------
	h5_path
		Path to ``kin_ion.h5``.
	band_start, band_stop
		0-based half-open band window; ``band_stop > band_start`` and
		``band_stop ≤ nb_total``.
	mesh
		Device mesh.  Required by ``SlabIOBackend.PHDF5_FFI``; the
		allgather backend tolerates ``None`` and returns a host-backed
		replicated JAX array.
	backend
		``SlabIOBackend`` selecting the underlying I/O path; defaults to
		the allgather backend.

	Returns
	-------
	jax.Array, shape ``(nk, nb, nb)``, dtype ``complex128``, replicated.
	"""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")

	# Peek shape so we can validate the slice and pass an explicit
	# (nk, nb, nb) request to read_slab — the FFI backend requires it.
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		nk, nb_total, nb_total2 = h5["kin_ion"].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"kin_ion must be square in band axes; got {(nk, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but kin_ion "
			f"only has {nb_total}"
		)

	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh, backend=backend) as io:
		arr = io.read_slab(
			"kin_ion",
			shape=(nk, nb, nb),
			offset=(0, band_start, band_start),
			dtype=jnp.complex128,
			mesh=mesh,
			partition_spec=P(None, None, None),
		)
	return arr

"""Kinetic + ionic Hamiltonian I/O.

The kin_ion matrix elements correspond to H_DFT − V_xc, i.e. kinetic +
ionic (local + nonlocal) contributions, with V_xc removed.

**Two modes, distinguished by the ``has_hartree`` attribute:**

``has_hartree=True`` (what ``gw.kin_ion_io`` writes by default)
    The dataset already contains ⟨mk|V_H|nk⟩, evaluated exactly on the
    FFT grid at generation time.  The GW driver MUST NOT add its own
    ISDF ``sig_h`` on top — see ``gw.sigma_dispatch``, which zeroes it,
    and ``gw.gw_output.write_results``, which skips it.  H₀ is then
    independent of the ISDF centroid count.

``has_hartree`` absent or False (legacy files)
    Kinetic + ionic only; the GW run adds the ISDF ``sig_h``.  H₀ then
    depends on the centroid basis through a ~500 eV cancellation, which
    ``_warn_on_unphysical_h0`` exists to catch.

Reading the attribute is the ONLY supported way to tell the two apart —
never infer it from magnitudes.
"""
import os

import h5py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .slab_io import SlabIO


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
	return out


def kin_ion_has_hartree(h5_path: str) -> bool:
	"""True iff ``kin_ion.h5`` already contains the mean-field V_H.

	The contract flag for the no-double-counting rule.  Legacy files
	written before the exact-V_H fold-in carry no attribute at all and
	therefore answer False, preserving their original semantics.
	"""
	try:
		attrs = read_kin_ion_provenance(h5_path)
	except (FileNotFoundError, KeyError):
		return False
	return bool(attrs.get("has_hartree", False))


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
	has_h = bool(attrs.get("has_hartree", False))
	if has_h:
		print_fn(
			"  kin_ion: has_hartree=True — exact FFT-grid V_H is already folded "
			f"in (truncation_2d={bool(attrs.get('hartree_truncation_2d', False))}); "
			"the ISDF sig_h will NOT be added."
		)
	else:
		print_fn(
			"  kin_ion: legacy mode (no has_hartree attr) — V_H comes from the "
			"ISDF centroid quadrature, so H0 depends on the centroid count."
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

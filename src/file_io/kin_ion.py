"""Kinetic + ionic Hamiltonian I/O.

The kin_ion matrix elements correspond to H_DFT - V_xc, i.e. kinetic + ionic
(local + nonlocal) contributions, with V_xc removed. The Hartree contribution
is not included unless explicitly added during kin_ion generation.
"""
import os

import h5py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .slab_io import SlabIO


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

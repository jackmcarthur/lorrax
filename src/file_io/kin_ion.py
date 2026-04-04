"""Kinetic + ionic Hamiltonian I/O.

The kin_ion matrix elements correspond to H_DFT - V_xc, i.e. kinetic + ionic
(local + nonlocal) contributions, with V_xc removed. The Hartree contribution
is not included unless explicitly added during kin_ion generation.
"""
import os
import h5py
import jax.numpy as jnp


def load_kin_ion_submatrix(h5_path: str, band_start: int, band_stop: int):
	"""Load sub-block of the kin_ion Hamiltonian for the requested band slice.

	The stored dataset is H_DFT - V_xc (kinetic + ionic; Hartree only if it was
	added at generation time), so V_xc should not be subtracted again later.
	
	Args:
		h5_path: Path to kin_ion.h5 file
		band_start: First band index (0-based)
		band_stop: One past last band index
	
	Returns:
		JAX array of shape (nk, nb, nb) with the Hamiltonian matrix elements
	"""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		kin_dset = h5["kin_ion"]
		nb_total = kin_dset.shape[1]
		if band_stop > nb_total:
			raise ValueError(
				f"Requested bands require {band_stop} states but kin_ion only has {nb_total}"
			)
		sub = kin_dset[:, band_start:band_stop, band_start:band_stop]
	return jnp.asarray(sub, dtype=jnp.complex128)

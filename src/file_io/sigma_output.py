"""Sigma self-energy output file writers."""
import os
import numpy as np
import h5py

from common.provenance import provenance_header


def write_sigma_to_file(
	sigma_sx_kij_eV,
	filename="eqp0.dat",
	sigma_coh_kij_eV=None,
	hartree_kij_eV=None,
	*,
	sx_label: str = "sigSX",
	corr_label: str = "sigCOH",
	total_label: str = "sigTOT",
):
	"""Write self-energy components to file.
	
	Args:
		sigma_sx_kij_eV: Exchange-like self-energy in eV, shape (nk, nb, nb)
		filename: Output file path
		sigma_coh_kij_eV: Correlation-like self-energy in eV, shape (nk, nb, nb)
		hartree_kij_eV: Hartree matrix elements in eV, shape (nk, nb, nb)
		sx_label: Text label for first self-energy column
		corr_label: Text label for second self-energy column
		total_label: Text label for the sum of first and second columns
	"""
	nk, nbands, _ = sigma_sx_kij_eV.shape

	# Resolve to absolute path, ensure directory exists
	abs_path = os.path.abspath(filename)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	with open(abs_path, "w") as f:
		# Write header with units
		f.write(provenance_header())
		f.write("# Sigma output (all in eV)\n")
		f.write(f"# {total_label} = {sx_label} + {corr_label}\n")
		for k in range(nk):
			f.write(f"\nk-point {k}:\n")
			f.write("-" * 100 + "\n")
			for n in range(nbands):
				sx_re = float(sigma_sx_kij_eV[k, n, n].real)
				sx_im = float(sigma_sx_kij_eV[k, n, n].imag)
				line = f"n={n:<3} {sx_label}={sx_re:>12.6f}"
				if abs(sx_im) > 1e-10:
					line += f"+{sx_im:>10.6f}i"
				else:
					line += "            "
				
				if sigma_coh_kij_eV is not None:
					coh_re = float(np.real(sigma_coh_kij_eV[k, n, n]))
					coh_im = float(np.imag(sigma_coh_kij_eV[k, n, n]))
					line += f"  {corr_label}={coh_re:>12.6f}"
					if abs(coh_im) > 1e-10:
						line += f"+{coh_im:>10.6f}i"
					else:
						line += "            "
					# Total COHSEX
					total_re = sx_re + coh_re
					total_im = sx_im + coh_im
					line += f"  {total_label}={total_re:>12.6f}"
					if abs(total_im) > 1e-10:
						line += f"+{total_im:>10.6f}i"
					else:
						line += "            "
				
				if hartree_kij_eV is not None:
					hv_re = float(np.real(hartree_kij_eV[k, n, n]))
					hv_im = float(np.imag(hartree_kij_eV[k, n, n]))
					line += f"  VH={hv_re:>12.6f}"
					if abs(hv_im) > 1e-10:
						line += f"+{hv_im:>10.6f}i"
				
				f.write(line + "\n")


def write_eqp_g0w0(
	eqp_path,
	energies_dft_ev,
	g0w0_diag_ev,
):
	"""Write E_DFT next to diagonal (H0 + Sigma_xc(E_DFT)) for G0W0 comparisons.

	Parameters
	----------
	eqp_path : str
		Output file path.
	energies_dft_ev : array (nk, nb)
		DFT eigenvalues in eV.
	g0w0_diag_ev : array (nk, nb)
		Diagonal matrix elements of (kin_ion + V_H + Sigma_xc(E_DFT)) in eV.
	"""
	energies_dft_ev = np.asarray(energies_dft_ev, dtype=np.float64)
	g0w0_diag_ev = np.asarray(g0w0_diag_ev, dtype=np.complex128)
	if energies_dft_ev.shape != g0w0_diag_ev.shape:
		raise ValueError(
			f"Shape mismatch for eqp_g0w0: DFT {energies_dft_ev.shape} vs G0W0 {g0w0_diag_ev.shape}"
		)

	abs_path = os.path.abspath(eqp_path)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	with open(abs_path, "w") as f:
		f.write(provenance_header())
		f.write("# G0W0 diagonal energies (eV)\n")
		f.write("# columns: band  E_DFT  Re[H0+Sigma_xc(E_DFT)]  Im[H0+Sigma_xc(E_DFT)]\n")
		for k in range(energies_dft_ev.shape[0]):
			f.write(f"\nk-point {k}:\n")
			f.write("-" * 80 + "\n")
			for n in range(energies_dft_ev.shape[1]):
				e_dft = float(energies_dft_ev[k, n])
				val = complex(g0w0_diag_ev[k, n])
				f.write(
					f"n={n:<3}  E_DFT={e_dft:>12.6f}  Re={val.real:>12.6f}  Im={val.imag:>12.6f}\n"
				)
	return abs_path


def write_sigma_freq_debug_table(
	filepath,
	*,
	energies_dft_ev,
	omega_rel_dft_ev,
	kin_ion_diag_ev,
	sigma_sex_static_diag_ev,
	sigma_coh_static_diag_ev,
	sigma_x_diag_ev,
	sigma_c_w0_diag_ev,
	sigma_c_plus_edft_ev,
	sigma_c_minus_edft_ev,
	sigma_c_invalid_static_diag_ev,
	sigma_c_edft_ev,
):
	"""Write per-(k,n) diagonal sigma decomposition used for GN-PPM debugging.

	All arrays are expected with shape (nk, nb). Values are written in eV.
	"""
	fields = [
		np.asarray(energies_dft_ev, dtype=np.float64),
		np.asarray(omega_rel_dft_ev, dtype=np.float64),
		np.asarray(kin_ion_diag_ev, dtype=np.complex128),
		np.asarray(sigma_sex_static_diag_ev, dtype=np.complex128),
		np.asarray(sigma_coh_static_diag_ev, dtype=np.complex128),
		np.asarray(sigma_x_diag_ev, dtype=np.complex128),
		np.asarray(sigma_c_w0_diag_ev, dtype=np.complex128),
		np.asarray(sigma_c_plus_edft_ev, dtype=np.complex128),
		np.asarray(sigma_c_minus_edft_ev, dtype=np.complex128),
		np.asarray(sigma_c_invalid_static_diag_ev, dtype=np.complex128),
		np.asarray(sigma_c_edft_ev, dtype=np.complex128),
	]
	ref_shape = fields[0].shape
	for arr in fields[1:]:
		if arr.shape != ref_shape:
			raise ValueError(f"sigma freq debug shape mismatch: {arr.shape} vs {ref_shape}")
	nk, nb = ref_shape

	abs_path = os.path.abspath(filepath)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	with open(abs_path, "w") as f:
		f.write(provenance_header())
		sep = "\t\t"
		col_w = 15
		cols = [
			"k",
			"n",
			"Edft-Ef",
			"E_dft",
			"kin_ion",
			"sex_0",
			"coh_0",
			"x_bare",
			"sig_c(0)",
			"sig_c+(w)",
			"sig_c-(w)",
			"sig_c_invld(0)",
			"sig_c(Edft)",
		]
		def _h(name: str) -> str:
			return f"{name:<{col_w}}"
		def _r(x: float) -> str:
			if np.isnan(x):
				return f"{'nan':>{col_w}}"
			return f"{x:>{col_w}.3f}"
		def _i(x: int) -> str:
			return f"{x:>{col_w}d}"
		f.write("# Sigma frequency debug decomposition (all energies in eV)\n")
		f.write("# " + sep.join(_h(c) for c in cols) + "\n")
		for ik in range(nk):
			f.write(f"\nk-point {ik}:\n")
			f.write(sep.join(_h(c) for c in cols) + "\n")
			for ib in range(nb):
				row = [
					_i(ik),
					_i(ib),
					_r(float(fields[1][ik, ib])),
					_r(float(fields[0][ik, ib])),
					_r(float(np.real(fields[2][ik, ib]))),
					_r(float(np.real(fields[3][ik, ib]))),
					_r(float(np.real(fields[4][ik, ib]))),
					_r(float(np.real(fields[5][ik, ib]))),
					_r(float(np.real(fields[6][ik, ib]))),
					_r(float(np.real(fields[7][ik, ib]))),
					_r(float(np.real(fields[8][ik, ib]))),
					_r(float(np.real(fields[9][ik, ib]))),
					_r(float(np.real(fields[10][ik, ib]))),
				]
				line = sep.join(row)
				f.write(line + "\n")
	return abs_path


def write_chunked_complex_dataset_h5(
	filepath,
	dataset_name,
	values,
	*,
	mode: str = "a",
	k_chunk_size: int = 16,
):
	"""Write a 3D/4D complex dataset to HDF5 in k-chunks."""
	abs_path = os.path.abspath(filepath)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	shape = tuple(values.shape)
	k_chunk = max(1, int(k_chunk_size))

	with h5py.File(abs_path, mode) as h5:
		if dataset_name in h5:
			del h5[dataset_name]
		if len(shape) == 4:
			n_omega, nk, nb, nb2 = shape
			dset = h5.create_dataset(
				dataset_name,
				shape=shape,
				dtype=np.complex128,
				chunks=(n_omega, min(k_chunk, nk), nb, nb2),
			)
			for k0 in range(0, nk, k_chunk):
				k1 = min(k0 + k_chunk, nk)
				dset[:, k0:k1, :, :] = np.asarray(values[:, k0:k1, :, :], dtype=np.complex128)
		elif len(shape) == 3:
			nk, nb, nb2 = shape
			dset = h5.create_dataset(
				dataset_name,
				shape=shape,
				dtype=np.complex128,
				chunks=(min(k_chunk, nk), nb, nb2),
			)
			for k0 in range(0, nk, k_chunk):
				k1 = min(k0 + k_chunk, nk)
				dset[k0:k1, :, :] = np.asarray(values[k0:k1, :, :], dtype=np.complex128)
		else:
			h5.create_dataset(dataset_name, data=np.asarray(values, dtype=np.complex128))

	return abs_path


def write_sigma_omega_h5(
	filepath,
	omega_ev,
	sigma_total_kij_ev=None,
	*,
	sigma_c_kij_ev=None,
	sigma_sx_kij_ev=None,
	hartree_kij_ev=None,
	k_chunk_size: int = 16,
	mesh=None,
	use_ffi_io: bool = False,
):
	"""Write frequency-dependent Sigma_mnk(omega) arrays to HDF5.

	Datasets:
	  - omega_ev: (n_omega,)                           — rank-0 metadata
	  - sigma_total_kij_ev: (n_omega, nk, nb, nb)      — large sharded
	  - sigma_c_kij_ev  (optional): (n_omega, nk, nb, nb)
	  - sigma_sx_kij_ev (optional): (nk, nb, nb)
	  - hartree_kij_ev  (optional): (nk, nb, nb)

	All large writes go through :mod:`file_io.slab_io`.  ``use_ffi_io``
	selects the backend: ``False`` (default) is the historical
	``process_allgather`` → rank-0 ``h5py`` path, consolidated into
	:class:`SlabIO`; ``True`` routes through the parallel-HDF5 FFI.
	"""
	from .slab_io import SlabIO

	abs_path = os.path.abspath(filepath)
	if sigma_total_kij_ev is None and sigma_c_kij_ev is None:
		raise ValueError("write_sigma_omega_h5 requires sigma_total_kij_ev or sigma_c_kij_ev.")

	if sigma_total_kij_ev is not None:
		shape_ref = tuple(sigma_total_kij_ev.shape)
	else:
		shape_ref = tuple(sigma_c_kij_ev.shape)
	if len(shape_ref) != 4:
		raise ValueError("dynamic sigma tensors must have shape (n_omega, nk, nb, nb).")
	n_omega, nk, nb, nb2 = shape_ref
	if nb != nb2:
		raise ValueError("dynamic sigma tensors must be square in band indices.")

	k_chunk = max(1, int(k_chunk_size))
	om_chunks  = (n_omega, min(k_chunk, nk), nb, nb2)
	kij_chunks = (min(k_chunk, nk), nb, nb2)

	# sigma_total_kij_ev is derived when not passed: total = c + sx + h
	total = sigma_total_kij_ev
	if total is None:
		total = sigma_c_kij_ev
		if sigma_sx_kij_ev is not None:
			total = total + sigma_sx_kij_ev[None, ...]
		if hartree_kij_ev is not None:
			total = total + hartree_kij_ev[None, ...]

	with SlabIO(abs_path, mode="w", mesh=mesh, use_ffi_io=use_ffi_io) as io:
		io.write_attr("omega_ev", np.asarray(omega_ev, dtype=np.float64))
		io.create_dataset("sigma_total_kij_ev",
			shape=shape_ref, dtype=np.complex128, chunks=om_chunks)
		io.write_slab("sigma_total_kij_ev", total,
			global_shape=shape_ref, k_chunk_size=k_chunk)
		if sigma_c_kij_ev is not None:
			io.create_dataset("sigma_c_kij_ev",
				shape=shape_ref, dtype=np.complex128, chunks=om_chunks)
			io.write_slab("sigma_c_kij_ev", sigma_c_kij_ev,
				global_shape=shape_ref, k_chunk_size=k_chunk)
		if sigma_sx_kij_ev is not None:
			io.create_dataset("sigma_sx_kij_ev",
				shape=tuple(sigma_sx_kij_ev.shape),
				dtype=np.complex128, chunks=kij_chunks)
			io.write_slab("sigma_sx_kij_ev", sigma_sx_kij_ev,
				global_shape=tuple(sigma_sx_kij_ev.shape),
				k_chunk_size=k_chunk)
		if hartree_kij_ev is not None:
			io.create_dataset("hartree_kij_ev",
				shape=tuple(hartree_kij_ev.shape),
				dtype=np.complex128, chunks=kij_chunks)
			io.write_slab("hartree_kij_ev", hartree_kij_ev,
				global_shape=tuple(hartree_kij_ev.shape),
				k_chunk_size=k_chunk)
	return abs_path


def copy_sigma_kij_h5_to_omega_h5(
	streamed_h5_path: str,
	output_path: str,
	omega_grid_ev,
	*,
	sigma_sx_kij_ev,
	hartree_kij_ev,
	omega_batch_size: int = 4,
):
	"""Convert a Ry-units streamed Σ_c(ω,k,i,j) HDF5 to an eV-units file.

	The ``_StreamedH5Accumulator`` path in ``ppm_sigma`` writes per-(τ × ω-batch)
	contributions into a transient ``sigma_c_kij_ry`` dataset (single-process
	only).  When the GW driver is in PPM mode and the accumulator returned
	``sigma_c_omega = None`` (streamed path), this helper copies the result
	into the canonical ``sigma_mnk.h5`` layout (eV units) with chunked reads
	so the host buffer never holds the full (n_omega × nk × nb × nb) tensor.

	Rank-0-only — caller must already gate.  Writes are serial h5py.
	"""
	import h5py
	from common.units import RYD_TO_EV

	with h5py.File(streamed_h5_path, "r") as h5_in:
		dset_c = h5_in["sigma_c_kij_ry"]
		n_omega, nk, nb, nb2 = dset_c.shape
		batch = max(1, min(int(omega_batch_size), n_omega))
		with h5py.File(output_path, "w") as h5_out:
			h5_out.create_dataset(
				"omega_ev",
				data=np.asarray(omega_grid_ev, dtype=np.float64),
			)
			dset_out = h5_out.create_dataset(
				"sigma_c_kij_ev",
				shape=dset_c.shape,
				dtype=np.complex128,
				chunks=(batch, max(1, min(4, nk)), nb, nb2),
			)
			h5_out.create_dataset(
				"sigma_sx_kij_ev",
				data=RYD_TO_EV * np.array(sigma_sx_kij_ev),
			)
			h5_out.create_dataset(
				"hartree_kij_ev",
				data=RYD_TO_EV * np.array(hartree_kij_ev),
			)
			for ibeg in range(0, n_omega, batch):
				iend = min(ibeg + batch, n_omega)
				dset_out[ibeg:iend] = (
					RYD_TO_EV
					* np.array(dset_c[ibeg:iend], dtype=np.complex128)
				)
	return output_path

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
	energies_dft_ev=None,
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
		energies_dft_ev: DFT (mean-field) band energies in eV, shape (nk, nb).
			When given, an ``Eo=`` column is written per row so the file can be
			aligned band-for-band against a BGW ``sigma_hp`` output (which lists
			``Eo``) — the mean-field energy is the reference-free key that ties
			LORRAX's band window to BGW's.
		sx_label: Text label for first self-energy column
		corr_label: Text label for second self-energy column
		total_label: Text label for the sum of first and second columns
	"""
	nk, nbands, _ = sigma_sx_kij_eV.shape

	# ------------------------------------------------------------------
	# REAL-VS-COMPLEX IS ONE DECISION PER COLUMN, TAKEN OVER THE WHOLE ARRAY
	# ------------------------------------------------------------------
	# This used to be decided PER SCALAR: each row asked whether its own
	# imaginary part exceeded 1e-10 and appended either "+x.xxxxxxi" or a
	# 12-space filler.  A run where one band's Sigma carries a numerically
	# significant imaginary part and its neighbours do not therefore emits
	# rows of two different SHAPES in one column — which is the malformed
	# dump seen in the frontier artifacts.  Two ways it bites: a
	# whitespace-splitting parser reads the next column's value into this
	# one, and the `VH` branch below had no filler at all, so the trailing
	# `Eo=` column moved horizontally depending on the row.
	#
	# A file format is a property of the FILE.  So the question is asked
	# once per component, over every element that will be written (the
	# diagonal of every k-point), and every row of that column then has the
	# same width whatever its own value happens to be.  A column that is
	# complex ANYWHERE is written complex EVERYWHERE, which is also the
	# honest report: "this quantity is complex in this run".
	#
	# Threshold unchanged (1e-10) and still absolute — this is a formatting
	# decision about eV-scale numbers, not a convergence test.
	_IM_TOL = 1e-10

	def _diag(a):
		"""The (nk, nbands) diagonal actually written, or None."""
		if a is None:
			return None
		return np.asarray(a)[:, np.arange(nbands), np.arange(nbands)]

	sx_diag = _diag(sigma_sx_kij_eV)
	coh_diag = _diag(sigma_coh_kij_eV)
	hv_diag = _diag(hartree_kij_eV)
	tot_diag = None if coh_diag is None else sx_diag + coh_diag

	def _is_complex(d):
		return d is not None and bool(np.any(np.abs(np.imag(d)) > _IM_TOL))

	sx_cplx = _is_complex(sx_diag)
	coh_cplx = _is_complex(coh_diag)
	tot_cplx = _is_complex(tot_diag)
	hv_cplx = _is_complex(hv_diag)

	#: Width of the omitted "+x.xxxxxxi" field, so a real column occupies
	#: exactly as many characters as a complex one would.
	_IM_PAD = " " * 12

	def _im(value, complex_column):
		"""The imaginary field for a column, or the exact-width filler.

		One function for both branches so the widths cannot drift apart:
		``+`` + 10 + ``i`` is 12 characters, and so is :data:`_IM_PAD`.
		"""
		if not complex_column:
			return _IM_PAD
		return f"+{float(value):>10.6f}i"

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
				sx_re = float(np.real(sx_diag[k, n]))
				sx_im = float(np.imag(sx_diag[k, n]))
				line = f"n={n:<3} {sx_label}={sx_re:>12.6f}"
				line += _im(sx_im, sx_cplx)

				if coh_diag is not None:
					coh_re = float(np.real(coh_diag[k, n]))
					coh_im = float(np.imag(coh_diag[k, n]))
					line += f"  {corr_label}={coh_re:>12.6f}"
					line += _im(coh_im, coh_cplx)
					# Total COHSEX
					total_re = sx_re + coh_re
					total_im = sx_im + coh_im
					line += f"  {total_label}={total_re:>12.6f}"
					line += _im(total_im, tot_cplx)

				if hv_diag is not None:
					hv_re = float(np.real(hv_diag[k, n]))
					hv_im = float(np.imag(hv_diag[k, n]))
					line += f"  VH={hv_re:>12.6f}"
					# Padded on the real branch too — it was NOT before, so
					# the trailing Eo= column moved row to row.
					line += _im(hv_im, hv_cplx)

				# Trailing Eo= column (mean-field energy) — appended last so
				# existing sigSX/sigCOH/sigTOT/VH parsers are unaffected.
				if energies_dft_ev is not None:
					line += f"  Eo={float(energies_dft_ev[k, n]):>12.6f}"

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
	filepath: str,
	columns: list[tuple[str, np.ndarray]],
) -> str:
	"""Write a per-(k, n) decomposition table.

	Each entry in ``columns`` is a ``(name, array)`` pair, where ``array``
	has shape ``(nk, nb)`` and is real or complex.  Real arrays produce one
	column; complex arrays produce two adjacent ``Re/Im`` sub-columns.  All
	values are assumed to already be in eV — the caller does the Ry→eV
	conversion at the seam (consistent with the rule "internals in Ry, eV
	only at print").

	The first row is a comment header naming all columns; subsequent rows
	are tab-separated numerical values, one per ``(k, n)`` pair, with k
	and n as the leading two integer columns.

	Returns
	-------
	The absolute path written.
	"""
	if not columns:
		raise ValueError("write_sigma_freq_debug_table: ``columns`` is empty.")

	arrays = [(name, np.asarray(arr)) for name, arr in columns]
	nk, nb = arrays[0][1].shape
	for name, arr in arrays:
		if arr.shape != (nk, nb):
			raise ValueError(
				f"column {name!r}: shape {arr.shape} != ({nk}, {nb})")

	# Column header — Re/Im split for complex arrays.
	header = ["k", "n"]
	for name, arr in arrays:
		if np.iscomplexobj(arr):
			header += [f"{name}.Re", f"{name}.Im"]
		else:
			header.append(name)

	abs_path = os.path.abspath(filepath)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	col_w = 16

	def _hdr(name: str) -> str:
		return f"{name:<{col_w}s}"

	def _val(x) -> str:
		if isinstance(x, complex) or np.iscomplexobj(x):
			# Should never reach here — complex columns are split above.
			x = float(np.real(x))
		fx = float(x)
		if np.isnan(fx):
			return f"{'nan':>{col_w}s}"
		return f"{fx:>+{col_w}.6f}"

	with open(abs_path, "w") as f:
		f.write(provenance_header())
		f.write(
			"# Sigma frequency debug decomposition (per-(k, n) diagonals; all "
			"energies in eV).\n"
		)
		f.write("# " + "\t".join(_hdr(h) for h in header) + "\n")
		for ik in range(nk):
			f.write(f"\nk-point {ik}:\n")
			for ib in range(nb):
				row = [f"{ik:>{col_w}d}", f"{ib:>{col_w}d}"]
				for name, arr in arrays:
					v = arr[ik, ib]
					if np.iscomplexobj(arr):
						row += [_val(v.real), _val(v.imag)]
					else:
						row.append(_val(v))
				f.write("\t".join(row) + "\n")

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


# ===========================================================================
# WHY sigma_mnk.h5 IS NOT STORED ON THE IBZ — measured 2026-08-08, REFUSED
# ===========================================================================
# The Σ cubes look like the ideal candidate for the store-on-IBZ /
# broadcast-on-read treatment ``kin_ion.h5`` now gets: pure output, one
# host-side consumer (``gw.eqp_bgw``), no convolution downstream, and half
# the payload sitting in two ``(n_omega, nk, nb, nb)`` tensors.  They are
# not, and the reason belongs here rather than being rediscovered.
#
# kin_ion IS computed on the IBZ, so its broadcast was write-time padding
# and moving it to the reader changes nothing.  Σ IS NOT.
# ``gw.sc_iteration`` says it in one line: "H/E/U on the IBZ, Σ on the full
# BZ".  Every full-BZ k in these cubes is an INDEPENDENT evaluation, so the
# k axis carries nk measurements — not nrk measurements and nk−nrk copies.
# Storing the wedge and rebuilding the rest replaces the other nk−nrk with
# reconstructions.
#
# MEASURED on ``tests/regression/cohsex_debug/sigma_mnk.h5`` (11.029 MB,
# nk 9, nrk 3 — a 3.00x reduction on this deck, not the 8x a Si 4³/48-op
# deck would suggest), residual of ``unfold(select(A))`` against ``A``:
#
#     dataset                rel Frobenius   worst element
#     hartree_kij_ev            3.0344e-01      113.08 eV
#     sigma_c_kij_ev            6.3020e-01        2.20 eV
#     sigma_sx_kij_ev           3.4971e-01        8.89 eV
#     sigma_total_kij_ev        3.0649e-01      105.27 eV
#     sigma_xc_qsgw_kij_ev      1.5653e-01        7.28 eV
#
# Those are not round-off.  The star relation is a PROPERTY a run has when
# its ISDF quadrature is orbit-closed, and cohsex_debug's 60-centroid set is
# not: ``tests/test_star_offdiag_gate.py`` measures the spatial pairs broken
# at 1.8e-01 … 4.0e-01 and the pure-TRS pairs holding only to
# 7.5e-07 … 7.0e-04.  The tree ALREADY refuses to select IBZ rows on numbers
# like these — ``sc_iteration._KSTAR_SPREAD_TOL`` is 1e-6 on Σ+V_H, five
# decades below what this fixture measures, and its refusal message is the
# argument in full: "members of a star must carry the same Σ up to
# round-off; they do not, so the full-BZ Σ and the IBZ carry are in
# different gauges and selecting the star representatives would silently
# keep the wrong one."
#
# And the spread is not noise to be squeezed out.  It is the METRIC the
# whole orbit-closure program is priced in (SPEC_qirr_restart_tensors §2a:
# Σ star spread 16.884 → 0.743 meV, a 23x gain, against +4.4 meV on BSE at
# fixed rank).  A format that reconstructs the star members deletes that
# measurement by construction, on every deck, closed or not.
#
# REGISTERED WHILE MEASURING, not claimed: ``sigma_total_kij_ev`` is 48.2 %
# of the payload and is the sum of the other three.  It reproduces
# ``c + sx[None] + h[None]`` to max|Δ| 4.351e-05 eV (4.440e-08 relative) but
# is NOT bit-identical under either association order, so dropping it today
# is a ~1.9x win that would move the file's numbers.  Pinning the writer's
# own association first would make it exact — a separate change with a
# separate gate, and the one actually worth doing on this file.


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
):
	"""Write frequency-dependent Sigma_mnk(omega) arrays to HDF5.

	Datasets:
	  - omega_ev: (n_omega,)                           — rank-0 metadata
	  - sigma_total_kij_ev: (n_omega, nk, nb, nb)      — large sharded
	  - sigma_c_kij_ev  (optional): (n_omega, nk, nb, nb)
	  - sigma_sx_kij_ev (optional): (nk, nb, nb)
	  - hartree_kij_ev  (optional): (nk, nb, nb)

	All large writes go through :mod:`file_io.slab_io`, which has one
	transport (per-rank collective MPI-IO) and no selector.
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

	with SlabIO(abs_path, mode="w", mesh=mesh) as io:
		io.write_attr("omega_ev", np.asarray(omega_ev, dtype=np.float64))
		io.create_dataset("sigma_total_kij_ev",
			shape=shape_ref, dtype=np.complex128, chunks=om_chunks)
		io.write_slab("sigma_total_kij_ev", total)
		if sigma_c_kij_ev is not None:
			io.create_dataset("sigma_c_kij_ev",
				shape=shape_ref, dtype=np.complex128, chunks=om_chunks)
			io.write_slab("sigma_c_kij_ev", sigma_c_kij_ev)
		if sigma_sx_kij_ev is not None:
			io.create_dataset("sigma_sx_kij_ev",
				shape=tuple(sigma_sx_kij_ev.shape),
				dtype=np.complex128, chunks=kij_chunks)
			io.write_slab("sigma_sx_kij_ev", sigma_sx_kij_ev)
		if hartree_kij_ev is not None:
			io.create_dataset("hartree_kij_ev",
				shape=tuple(hartree_kij_ev.shape),
				dtype=np.complex128, chunks=kij_chunks)
			io.write_slab("hartree_kij_ev", hartree_kij_ev)
	return abs_path

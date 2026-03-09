"""Sigma self-energy output file writers."""
import os
import numpy as np


def write_sigma_to_file(sigma_sx_kij_eV, filename="eqp0.dat", sigma_coh_kij_eV=None, hartree_kij_eV=None):
	"""Write COHSEX self-energy components to file.
	
	Args:
		sigma_sx_kij_eV: Screened exchange self-energy in eV, shape (nk, nb, nb)
		filename: Output file path
		sigma_coh_kij_eV: Coulomb hole self-energy in eV, shape (nk, nb, nb)
		hartree_kij_eV: Hartree matrix elements in eV, shape (nk, nb, nb)
	"""
	nk, nbands, _ = sigma_sx_kij_eV.shape

	# Resolve to absolute path, ensure directory exists
	abs_path = os.path.abspath(filename)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	with open(abs_path, "w") as f:
		# Write header with units
		f.write("# COHSEX output: Sigma_SX (screened exchange), Sigma_COH (Coulomb hole), VH (Hartree) - all in eV\n")
		f.write("# Sigma_COHSEX = Sigma_SX + Sigma_COH\n")
		for k in range(nk):
			f.write(f"\nk-point {k}:\n")
			f.write("-" * 100 + "\n")
			for n in range(nbands):
				sx_re = float(sigma_sx_kij_eV[k, n, n].real)
				sx_im = float(sigma_sx_kij_eV[k, n, n].imag)
				line = f"n={n:<3} sigSX={sx_re:>12.6f}"
				if abs(sx_im) > 1e-10:
					line += f"+{sx_im:>10.6f}i"
				else:
					line += "            "
				
				if sigma_coh_kij_eV is not None:
					coh_re = float(np.real(sigma_coh_kij_eV[k, n, n]))
					coh_im = float(np.imag(sigma_coh_kij_eV[k, n, n]))
					line += f"  sigCOH={coh_re:>12.6f}"
					if abs(coh_im) > 1e-10:
						line += f"+{coh_im:>10.6f}i"
					else:
						line += "            "
					# Total COHSEX
					total_re = sx_re + coh_re
					total_im = sx_im + coh_im
					line += f"  sigTOT={total_re:>12.6f}"
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


def write_eqp1(
	eqp1_path,
	energies_dft_ev,
	energies_qp_ev,
	E_oneshot_ev,
	nkx, nky, nkz,
	nb_sigma,
):
	"""Write eqp1.dat with DFT, one-shot, and QP eigenvalues per k-point.

	Parameters
	----------
	eqp1_path : str
		Output file path.
	energies_dft_ev : array (nk, nb)
		DFT eigenvalues in eV.
	energies_qp_ev : array (nk, nb)
		QP eigenvalues in eV (from diagonalizing H_DFT + Σ).
	E_oneshot_ev : array (nk, nb)
		One-shot diagonal energies in eV.
	nkx, nky, nkz : int
		k-grid dimensions.
	nb_sigma : int
		Number of bands in sigma window.
	"""
	energies_dft_ev = np.asarray(energies_dft_ev, dtype=np.float64)
	energies_qp_ev = np.asarray(energies_qp_ev, dtype=np.float64)
	E_oneshot_ev = np.asarray(E_oneshot_ev, dtype=np.float64)

	abs_path = os.path.abspath(eqp1_path)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)

	with open(abs_path, "w") as f:
		f.write("# kx ky kz nbands\n")
		f.write("# spin band E_DFT E_oneshot(DFT-basis) E_QP(eigh)\n")
		ik = 0
		for ikz in range(nkz):
			for iky in range(nky):
				for ikx in range(nkx):
					kx = ikx / nkx
					ky = iky / nky
					kz = ikz / nkz
					f.write(f"  {kx:.9f}  {ky:.9f}  {kz:.9f}      {nb_sigma}\n")
					for ib in range(nb_sigma):
						e_dft = float(energies_dft_ev[ik, ib])
						e_oneshot = float(E_oneshot_ev[ik, ib])
						e_qp = float(energies_qp_ev[ik, ib])
						f.write(f"       1       {ib+1}  {e_dft:14.9f}  {e_oneshot:14.9f}  {e_qp:14.9f}\n")
					ik += 1
	return abs_path


def write_eqp_table(dft_energies_ev, qp_energies_ev, filename="eqp.dat"):
	"""Write DFT vs quasiparticle energies per (k,n) in eqp-style text format.

	Args:
		dft_energies_ev: array-like with shape (nk, nb)
		qp_energies_ev: array-like with shape (nk, nb)
		filename: output path for the table
	"""
	dft = np.asarray(dft_energies_ev, dtype=np.float64)
	qp = np.asarray(qp_energies_ev, dtype=np.complex128)
	if dft.shape != qp.shape:
		raise ValueError(f"Shape mismatch for EQP table: DFT {dft.shape} vs QP {qp.shape}")

	abs_path = os.path.abspath(filename)
	dirname = os.path.dirname(abs_path)
	if dirname:
		os.makedirs(dirname, exist_ok=True)
	print(f"Writing EQP table to: {abs_path}")

	nk, nb = dft.shape
	with open(abs_path, "w") as f:
		for k in range(nk):
			f.write(f"\nk-point {k}:\n")
			f.write("-" * 60 + "\n")
			for n in range(nb):
				e_dft = float(dft[k, n])
				e_qp = complex(qp[k, n])
				f.write(
					f"n={n:<3} EDFT={e_dft:>9.4f}  EQP={e_qp.real:>9.4f} + {e_qp.imag:>9.4f}i\n"
				)


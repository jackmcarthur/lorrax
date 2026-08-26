"""Sigma self-energy output file writers."""
import os
import numpy as np
import h5py

from common.provenance import provenance_header

# THE STAMP CONTRACT IS ``kin_ion``'s, IMPORTED RATHER THAN RESTATED.
# ``sigma_mnk.h5`` and ``kin_ion.h5`` now both store a k axis that may be
# the irreducible wedge, and they must mean the SAME thing by that or a
# reader will eventually apply one file's rule to the other.  So the attr
# names, the version integer, the two table names and the
# "no attr means full" default all come from one module, and
# :func:`kin_ion.read_star_map` is the one implementation that reads them.
from .kin_ion import (                                      # noqa: F401
	IRR_IDX_DATASET,
	K_STORAGE_ATTR,
	K_STORAGE_FULL,
	K_STORAGE_IBZ,
	K_STORAGE_VERSION,
	K_STORAGE_VERSION_ATTR,
	N_SYM_SPATIAL_ATTR,
	SYM_IDX_DATASET,
	read_star_map,
)

#: Attr prefix under which the pre-extraction star-spread measurement is
#: stamped.  Four numbers per dataset — see :func:`sigma_star_spread_stats`
#: for what each one is and why four rather than one.
SPREAD_ATTR_PREFIX = "star_spread_"

#: Which datasets of ``sigma_mnk.h5`` carry a k axis, and where it is.
#: The dynamic cubes are ``(n_omega, nk, nb, nb)``; everything else is
#: ``(nk, …)``.  A dataset absent from this table has no k axis at all
#: (``omega_ev``) and is written verbatim.
SIGMA_K_AXIS = {
	"sigma_total_kij_ev": 1,
	"sigma_c_kij_ev": 1,
	"sigma_sx_kij_ev": 0,
	"hartree_kij_ev": 0,
	"sigma_xc_qsgw_kij_ev": 0,
	"qp_diag_self_consistent_ev": 0,
	"qp_omega0_ev": 0,
	"qp_static_cohsex_ev": 0,
	# The Σ_c band-convergence fit, band-DIAGONAL (nk, nb), so k is axis 0.
	# Registered here and not only written, because that is what puts them
	# through the same star extraction and the same star-spread instrument
	# as the cubes: the invariant that matters for S_inf is exact star
	# covariance, and it is only checkable on a persisted, stamped array.
	"sigma_c_extrap_inf_kn_ev": 0,
	"sigma_c_extrap_last_kn_ev": 0,
	"sigma_c_extrap_ampl_kn_ev": 0,
	"sigma_c_extrap_sigma_kn_ev": 0,
	# The estimator-specific fourth array.  ``ampl`` is A (the 1/N
	# coefficient) and is written by ``band_index_only``; ``beta`` is the
	# per-state decay exponent and is written by ``spectral_shell``.  Two
	# names rather than one reused name, so a file says which estimator made
	# it even to a reader who never looks at the attributes.
	"sigma_c_extrap_beta_kn": 0,
	# The energies THIS Sigma was evaluated at, omega-relative, band
	# diagonal (nk, nb) -> k is axis 0.  Registered here rather than merely
	# written, so it rides the same star extraction and the same star-spread
	# instrument as the cubes it describes: an evaluation spectrum that is
	# not star-covariant would mean the cube's rows and its stamp disagree
	# about which k they are.
	"sigma_eval_rel_ev": 0,
}

#: The ω axis, and the two attrs on it that say what it is measured FROM.
#: ``omega_ev`` has always been a RELATIVE axis; nothing in the file said
#: relative to what, so every post-hoc consumer guessed, and the one that
#: guessed insulating midgap mis-sampled Σ_c(ω) by a measured 2.79 eV on
#: the sodium metal deck (audit A2).  One name, defined once, used by the
#: writer and by :func:`read_omega_reference`.
OMEGA_DATASET = "omega_ev"
OMEGA_REFERENCE_ATTR = "omega_reference_ev"
OMEGA_REFERENCE_PROVENANCE_ATTR = "omega_reference_provenance"

#: The EFFECTIVE Sigma broadening the run used, and the value the deck
#: asked for.  Stamped on the same axis as the reference above and for the
#: same reason: the deck key `sigma_regularization_ev` is a REQUEST, an
#: ansatz may raise it to a conditioning floor, and until this stamp
#: existed the resolved value lived only in a print -- so a cross-ansatz
#: comparison could not assert that two runs shared xi, only assume it.
SIGMA_REGULARIZATION_ATTR = "sigma_regularization_ev"
SIGMA_REGULARIZATION_REQUESTED_ATTR = "sigma_regularization_requested_ev"
SIGMA_REGULARIZATION_FLOOR_POLICY_ATTR = "sigma_regularization_floor_policy"

#: The provenance values the driver stamps — ONE per ``fermi_reference``
#: value, and they are produced by ``gw.efermi.resolve_sigma_efermi_ry``
#: rather than inferred here.  ``fixed-N mu`` is the metal path's chemical
#: potential from the fixed-N MP1 solve; ``midgap`` is the insulating
#: loader convention (``wfn.efermi``, ½(VBM+CBM)); ``vbm`` is
#: ``fermi_reference = vbm``, which used to be indistinguishable from
#: ``midgap`` in this file because the stamp was derived from "did the
#: caller pass an explicit reference" rather than from the deck key.
OMEGA_REFERENCE_FIXED_N_MU = "fixed-N mu"
OMEGA_REFERENCE_MIDGAP = "midgap"
OMEGA_REFERENCE_VBM = "vbm"

#: WHERE this Sigma was evaluated, stamped on :data:`SIGMA_EVAL_DATASET`.
#: A DIFFERENT fact from the omega reference above: that says what the axis
#: is measured FROM, this says which spectrum the cube was built AT.  Until
#: 2026-08-22 the file carried the first and not the second, so
#: ``gw.eqp_bgw.make_eqp_bgw`` had no way to tell a one-shot cube from a
#: self-consistent one and silently reverted to the at-E_DFT linearization
#: -- correct for one-shot, a different calculation for SC.
SIGMA_EVAL_DATASET = "sigma_eval_rel_ev"
SIGMA_EVAL_PROVENANCE_ATTR = "sigma_eval_provenance"
SIGMA_EVAL_AT_E_DFT = "at_e_dft"
SIGMA_EVAL_SELF_CONSISTENT = "self_consistent_qp"

#: How much of the evaluation spectrum the omega grid actually SAMPLED, and
#: what was done with the rest.  Stamped beside the eval energies because a
#: consumer that centres on them needs to know which of them the cube can
#: answer for: the Na semicore run wrote 41.3 % endpoint values as if they
#: were Sigma at the state's own energy.
OMEGA_COVERAGE_N_ATTR = "omega_uncovered_count"
OMEGA_COVERAGE_FRAC_ATTR = "omega_uncovered_fraction"
OMEGA_COVERAGE_POLICY_ATTR = "omega_out_of_range_policy"

#: The datasets :func:`write_sigma_omega_h5` creates the file with.  Every
#: run that writes ``sigma_mnk.h5`` at all writes these, so the appender
#: below can use any one of them that is present to learn the file's own k
#: storage rather than being told it.
SIGMA_CUBE_DATASETS = (
	"sigma_total_kij_ev",
	"sigma_c_kij_ev",
	"sigma_sx_kij_ev",
	"hartree_kij_ev",
)

#: The datasets the OPT-IN ``write_qsgw_datasets`` deck key adds, in the
#: order a reader meets them.  They are the plotting payload: the static
#: Hermitian Σ_xc the QSGW ansatz produces, and the QP energy ladders of
#: the approximations one wants to see beside each other.  Names and
#: shapes are the ones the pre-2026-04 driver wrote and the committed
#: ``cohsex_debug`` fixture still holds, so a script written against that
#: file keeps working against a new one.
QSGW_PLOT_DATASETS = (
	"sigma_xc_qsgw_kij_ev",
	"qp_static_cohsex_ev",
	"qp_omega0_ev",
	"qp_diag_self_consistent_ev",
)


def write_sigma_to_file(
	sigma_sx_kij_eV,
	filename="eqp0.dat",
	sigma_coh_kij_eV=None,
	hartree_kij_eV=None,
	energies_dft_ev=None,
	*,
	kpoints_crys,
	star_spread_ev=None,
	star_spread_per_band_ev=None,
	star_spread_multiplet_ev=None,
	n_star_members=None,
	sx_label: str = "sigSX",
	corr_label: str = "sigCOH",
	total_label: str = "sigTOT",
):
	"""Write self-energy components to file.

	Args:
		sigma_sx_kij_eV: Exchange-like self-energy in eV, shape (nk, nb, nb)
		filename: Output file path
		kpoints_crys: (nk, 3) crystal coordinates, ONE ROW PER Sigma ROW.
			REQUIRED, and checked against ``nk``.  See the k-BASIS note
			below: this file's rows are whatever basis the caller's Sigma
			is on, and this argument is how the file says which.
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

	k-BASIS — WHY EVERY BLOCK CARRIES ITS COORDINATE
	------------------------------------------------
	The block label ``k-point N`` is a POSITION in whatever array the
	caller handed in, and nothing in the file used to say which k that
	was.  Three separate downstream comparisons then paired this file's
	rows against BerkeleyGW's IBZ blocks by position, because positions
	0,1,2 coincide on Si 4x4x4 and only diverge from row 3 on (the true
	map is ``[0,1,2,5,6,7,10,27]``).  The worst of them reported a
	291 meV disagreement where the real figure was 28 meV; an earlier
	one manufactured a 600 meV "non-symmorphic phase bug".

	So the coordinate is written on every block, and ``kpoints_crys`` is
	REQUIRED and length-checked against the Sigma rows.  A consumer can
	now join on the coordinate instead of the position, and a reader can
	see at a glance which basis the file is on -- an IBZ file and a
	full-BZ file no longer look identical.

	This writer does NOT decide the basis; ``gw_output.write_results``
	does, and since 2026-08-15 it writes EVERY text file on the
	irreducible wedge — the k-set Sigma is actually extracted on.  The
	full-BZ arrays are that wedge's symmetry image and carry no
	independent information, so writing them was writing the same numbers
	several times under indices nothing in the file explained.
	A consumer that genuinely needs the full BZ unfolds through the
	symmetry service (``symmetry_maps.star_broadcast``, via
	``file_io.kin_ion.broadcast_ibz_to_full_bz``), which is what
	``htransform.read_eqp_energies`` and ``bse_io.apply_eqp_corrections``
	do.  Nothing reconstructs a star by hand.

	The coordinate goes on its OWN line, never appended to the
	``k-point N:`` header and never onto a data row.  Both of those
	placements break parsers in tree: htransform anchors its header
	regex with ``$`` (``htransform.py:1153``), and six parsers
	discriminate header-from-body by "exactly four whitespace tokens".
	A separate ``#`` line is invisible to every one of them, exactly as
	the existing ruler line already is.
	"""
	nk, nbands, _ = sigma_sx_kij_eV.shape
	kpts = np.asarray(kpoints_crys, dtype=np.float64)
	if kpts.shape != (nk, 3):
		# The structural guard: handing 64 rows of full-BZ Sigma an
		# 8-row IBZ k-list (or vice versa) is the mistake this file
		# exists to make impossible, so it is refused rather than
		# written out and discovered downstream.
		raise ValueError(
			f"kpoints_crys shape {kpts.shape} does not match the {nk} "
			f"Sigma k-rows of {os.path.basename(str(filename))} — the "
			f"k-list and the self-energy are on different k-bases.")

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
		f.write(f"# k-basis: irreducible wedge, {nk} k-points; each block "
		        f"states its crystal coordinate on a '# kcrys' line\n")
		# The star-spread diagnostic, MEASURED ON THE FULL BZ upstream (see
		# ``gw_output._star_spread_of_sigma_diag``) because a wedge file
		# cannot carry it: unfolding the wedge back is a gather, so the
		# spread would read 0.000 by construction.  Recorded here so the
		# check survives the file's move to the wedge.
		if star_spread_ev is not None:
			f.write(f"# star_spread_ev {float(star_spread_ev):.9e}   "
			        f"# max over ALL {nbands} bands of the per-band max-min "
			        f"of Re diag {total_label} within one star, over the "
			        f"{int(n_star_members)} full-BZ k this wedge unfolds "
			        f"to.  READ THE PER-BAND ROW BELOW, NOT THIS, IF YOU "
			        f"COMPARE A BAND SUBSET: the band scope belongs to the "
			        f"consumer, and this max answers the widest possible "
			        f"question.  BLIND TO THE TRS CONJUGATION CLASS by "
			        f"construction (conjugating a Hermitian block leaves "
			        f"its real diagonal intact) — that question is gated in "
			        f"tests/test_star_offdiag_gate.py\n")
		if star_spread_per_band_ev is not None:
			_pb = np.asarray(star_spread_per_band_ev, dtype=np.float64)
			if _pb.shape != (nbands,):
				raise ValueError(
					f"star_spread_per_band_ev has shape {_pb.shape}, "
					f"expected ({nbands},) — one entry per written band.")
			f.write("# star_spread_ev_per_band "
			        + " ".join(f"{float(v):.9e}" for v in _pb) + "\n")
		if star_spread_multiplet_ev is not None:
			# THE DEGENERACY-RESOLVED TWIN, and it is not a refinement of the
			# row above -- it answers a question that one cannot.  A single
			# band inside a degenerate multiplet has no symmetry-invariant
			# Re Sigma_bb: any unitary mixing within the subspace is an
			# equally valid eigenbasis, so the per-band row measures the
			# eigensolver's gauge there as much as the physics.  The TRACE
			# over the multiplet IS invariant.  Entries are per band (the
			# subspace spread divided by its size) so the two rows compare
			# element by element; where a band is isolated the two agree by
			# construction.  See gw_output._star_spread_over_multiplets.
			_mp = np.asarray(star_spread_multiplet_ev, dtype=np.float64)
			if _mp.shape != (nbands,):
				raise ValueError(
					f"star_spread_multiplet_ev has shape {_mp.shape}, "
					f"expected ({nbands},) — one entry per written band.")
			# NAMED "star_spread_multiplet_ev*", NOT "star_spread_ev_*_multiplet".
			# "# star_spread_ev" is already a PREFIX of
			# "# star_spread_ev_per_band", and any reader that counts rows by
			# prefix -- tests/test_eqp_kpoint_basis.py does -- breaks the
			# moment a third key extends that stem.  Measured: the first
			# spelling of this row did exactly that.  A new header key must
			# not extend an existing one.
			f.write(f"# star_spread_multiplet_ev {float(_mp.max()):.9e}\n")
			f.write("# star_spread_multiplet_ev_per_band "
			        + " ".join(f"{float(v):.9e}" for v in _mp) + "\n")
		for k in range(nk):
			f.write(f"\nk-point {k}:\n")
			f.write(f"# kcrys {kpts[k, 0]:15.9f}{kpts[k, 1]:15.9f}"
			        f"{kpts[k, 2]:15.9f}\n")
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
	*,
	kpoints_crys,
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
	kpoints_crys : array (nk, 3)
		Crystal coordinates, one row per energy row.  REQUIRED and
		length-checked, for the reason spelled out in the k-BASIS note
		of :func:`write_sigma_to_file` — this file has the same
		``k-point N:`` block layout and was paired by position by the
		same downstream scripts.  On the irreducible wedge, like every
		other text file ``gw_output.write_results`` emits; its one
		former full-BZ consumer was the out-of-tree
		``make_eqp_htformat.py``, whose whole job was pre-unfolding this
		file for htransform — which now reads the wedge and unfolds
		through the symmetry service itself.
	"""
	energies_dft_ev = np.asarray(energies_dft_ev, dtype=np.float64)
	g0w0_diag_ev = np.asarray(g0w0_diag_ev, dtype=np.complex128)
	if energies_dft_ev.shape != g0w0_diag_ev.shape:
		raise ValueError(
			f"Shape mismatch for eqp_g0w0: DFT {energies_dft_ev.shape} vs G0W0 {g0w0_diag_ev.shape}"
		)
	kpts = np.asarray(kpoints_crys, dtype=np.float64)
	if kpts.shape != (energies_dft_ev.shape[0], 3):
		raise ValueError(
			f"kpoints_crys shape {kpts.shape} does not match the "
			f"{energies_dft_ev.shape[0]} energy k-rows of eqp_g0w0 — the "
			f"k-list and the energies are on different k-bases.")

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
			f.write(f"# kcrys {kpts[k, 0]:15.9f}{kpts[k, 1]:15.9f}"
			        f"{kpts[k, 2]:15.9f}\n")
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
	*,
	kpoints_crys,
) -> str:
	"""Write a per-(k, n) decomposition table.

	``kpoints_crys`` is (nk, 3) crystal coordinates, REQUIRED and
	length-checked against the columns, and written as a ``# kcrys`` line
	under each ``k-point N:`` header — the same rule the other two text
	writers in this module follow, for the same reason.  This file carries
	``k`` as a DATA COLUMN, which makes it look self-describing and is
	exactly the trap: that integer is a position in whatever array the
	caller passed, and pairing it against another code's k-order by value
	is the mistake that produced 291 meV and 600 meV phantom
	disagreements elsewhere in this tree.  The coordinate is the join key;
	the integer is not.

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
	kpts = np.asarray(kpoints_crys, dtype=np.float64)
	if kpts.shape != (nk, 3):
		raise ValueError(
			f"kpoints_crys shape {kpts.shape} does not match the {nk} k-rows "
			f"of {os.path.basename(str(filepath))} — the k-list and the "
			f"columns are on different k-bases.")

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
			f.write(f"# kcrys {kpts[ik, 0]:15.9f}{kpts[ik, 1]:15.9f}"
			        f"{kpts[ik, 2]:15.9f}\n")
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


# ===========================================================================
# HOW sigma_mnk.h5 STORES ITS k AXIS — the SELECTION, and what it is not
# ===========================================================================
# This file can store its k axis on the irreducible wedge, and the way it
# does so is a SELECTION of rows that were already accumulated: the Σ sweep
# runs at every full-BZ k exactly as before, the cube is completed, the
# kernel exits, and only then does the writer keep one row per star and drop
# the rest.  Nothing is reconstructed, nothing is substituted, and the rows
# that survive are bit-for-bit the rows the full-BZ file held.
#
# THAT IS NOT THE THING THIS FILE REFUSED, and the distinction is the whole
# design.  What was refused — and stays refused — is storing the wedge and
# REBUILDING the other nk−nrk rows from it on read.  ``gw.sc_iteration``
# says why in one line: "H/E/U on the IBZ, Σ on the full BZ".  Every full-BZ
# k is an INDEPENDENT evaluation, so an unfold would replace nk−nrk
# measurements with reconstructions, and on a deck whose quadrature is not
# orbit-closed those reconstructions are wrong by the amounts below.  No
# reader in this module unfolds; a consumer that wants full-BZ Σ rows must
# get them from a run, not from this file.
#
# The reason a selection is nevertheless sound is that nothing downstream
# consumes the dropped rows.  EQP consumption (kin_ion + v_hartree + Σ^xc)
# is already k_irr-side, and the static Hermitianization now happens on the
# selected rows only, so the extraction removes payload that had no reader.
#
# WHAT THE DROPPED ROWS USED TO LET YOU MEASURE, and where it went.  Because
# the star relation is the metric the whole orbit-closure program is priced
# in (SPEC_qirr_restart_tensors §2a: Σ star spread 16.884 → 0.743 meV, a 23x
# gain, against +4.4 meV on BSE at fixed rank), an artifact that no longer
# carries the star members cannot be measured for it afterwards.  So the
# measurement moved to WRITE time: :func:`sigma_star_spread_stats` runs on
# the complete full-BZ cube before a single row is dropped, and its four
# numbers are logged and stamped onto every extracted dataset.  The metric
# survives the extraction; only the ability to recompute it from the file
# does not.
#
# THE NUMBERS, and what they turned out to be.  Measured on
# ``tests/regression/cohsex_debug/sigma_mnk.h5`` (nk 9, nrk 3 — a 3.00x
# reduction on this deck, not the 8x a Si 4³/48-op deck would suggest),
# residual of ``unfold(select(A))`` against ``A``:
#
#     dataset                worst element   Re-diag spread   after degen-avg
#     hartree_kij_ev            113.08 eV         61.51 eV         56.76 eV
#     sigma_total_kij_ev        105.27 eV         59.53 eV         54.75 eV
#     sigma_sx_kij_ev             8.89 eV          1.21 eV          1.16 eV
#     sigma_xc_qsgw_kij_ev        7.28 eV          4.30 eV          4.20 eV
#     sigma_c_kij_ev              1.43 eV          0.98 eV          0.97 eV
#
# The 113.08 eV was investigated before the extraction was written, on the
# owner's instruction, because a number that large usually means the Σ
# procedure did something wrong.  It did not, and the answer is in two
# halves (``tools/sigma_star_spread_decompose.py``, and the prose in
# ``NOTE_sigma_star_spread.md``).
#
# HALF ONE: the raw metric can read enormous from gauge alone.  ⟨mk|O|nk⟩
# is not a physical quantity when m and n are degenerate — the diagonaliser
# picks a basis inside each manifold independently at every k.  ``kin_ion``,
# built on the exact FFT grid with no ISDF quadrature anywhere in its path,
# is the control: it reads 2.861e-01 relative and 7.78 Ry on its worst
# element under this metric, and 6.4e-13 under one the gauge cannot move.
# The 113.08 eV element itself sits at (k, m, n) = (8, 0, 6), OFF-DIAGONAL,
# stored −52.7533+20.3097i against an unfold of +52.7210−20.4598i: the
# magnitudes are 56.5278 and 56.5518, agreeing to 4 parts in 10⁴, with the
# phase inverted, and band 0 lies in a manifold of multiplicity 2.  A 56 eV
# element reports 113 eV the moment the sign convention differs.
#
# HALF TWO: what is left after the gauge is removed is real, and it is this
# fixture's quadrature.  Degeneracy-averaging the Hartree diagonal takes its
# star spread only from 61.51 eV to 56.76 eV — 7.7 % was gauge.
# ``verify_centroid_orbit_closure`` on this deck's own
# ``centroids_frac_60.txt`` returns the identity op at exactly 0.0000 and
# all eleven genuine rotations at 0.166 … 0.276, and the Σ spread has the
# same signature: relations needing a rotation break at 8e-02 … 3e-01
# gauge-blind, relations needing none hold at 1.1e-04 (hartree) and 2.6e-04
# (sigma_sx).  Cause and effect line up op by op.
#
# NONE OF WHICH THE SELECTION TOUCHES.  A run whose quadrature is not
# orbit-closed has a Σ that differs across a star, and that was true of the
# full-BZ file too; the extraction neither creates the disagreement nor
# hides it, because the stamped stats report it on every run.  What the
# extraction does mean is that the file no longer offers a CHOICE of which
# star member to believe — it keeps the first, which is the row every
# k_irr-side consumer was already reading.
#
# CORRECTED WHILE MEASURING.  ``sigma_total_kij_ev`` reproduces
# ``c + sx[None] + h[None]`` on that fixture to max|Δ| 4.3511e-05 eV, and
# this block used to attribute the gap to an unpinned association order.  It
# is not that.  All three association orders give BIT-IDENTICALLY the same
# 4.3511e-05 eV, so reassociation is not what moves it; every one of the four
# arrays is exactly a float32 value widened into a complex128 container, and
# 4.3511e-05 against a peak of 384.51 eV is 1.13e-07 relative, which is
# float32 epsilon.  The fixture is a single-precision artifact.  Pinning the
# association would not have made it exact — and the writer's two paths
# already share one association order by construction, ``(c + sx) + h``, in
# both the replicated derivation below and ``gw.ppm_pipeline``'s sharded
# one.  What was missing was a gate saying so, and
# ``tests/test_sigma_kirr_extraction.py`` is now it.


def compact_star_tables(irr_idx_k):
	"""``(rows_to_keep, compacted_irr_idx_k)`` for a star selection.

	``rows_to_keep`` are the FIRST-OCCURRENCE full-BZ rows of each star,
	in that order, which is ``star_select``'s own convention and therefore
	the ordering ``star_broadcast`` expects to be handed back.

	THE COMPACTION IS NOT COSMETIC.  ``SymMaps.irr_idx_k`` labels each
	full-BZ k with a row of the WFN's wedge, and that wedge can have more
	rows than the mesh has stars — ``cohsex_debug`` is exactly that case,
	with ``irr_idx_k`` taking the values ``{0, 2, 3}`` out of a 4-row
	wedge for 3 stars.  Written verbatim, such a table claims 4 stored
	rows for a slab that has 3, which is precisely the inconsistency
	:func:`kin_ion.read_star_map` refuses on.  So the stored table is
	renumbered to index the STORED rows, and the file is self-consistent
	by construction rather than by the caller's luck.
	"""
	irr = np.asarray(irr_idx_k)
	if irr.ndim != 1:
		raise ValueError(
			f"irr_idx_k must be (nk_full,); got shape {irr.shape}")
	seen: dict[int, int] = {}
	rows: list[int] = []
	compact = np.empty(irr.size, dtype=np.int32)
	for i, lab in enumerate(int(v) for v in irr):
		if lab not in seen:
			seen[lab] = len(rows)
			rows.append(i)
		compact[i] = seen[lab]
	return np.asarray(rows, dtype=np.int64), compact


def star_select_k_irr(values, rows_to_keep, *, k_axis=0):
	"""Keep ``rows_to_keep`` along ``k_axis``.  Pure selection, no arithmetic.

	This is the whole extraction.  It is spelled as its own function so the
	gate can assert what it is: a take of rows that already existed, which
	makes bit-identity against the full-BZ array trivially true rather
	than something to be measured to a tolerance.

	A DEVICE ARRAY STAYS ON ITS DEVICE.  ``jax.Array`` carries ``.take``,
	so the sharded write path drops its rows without a host round trip of
	the cube — the k axis is never the sharded one here (the sharded
	layout tiles bands, ``qsgw_utils.is_band_sharded_sigma_omega``), so
	this is a local gather on every rank.
	"""
	rows = np.asarray(rows_to_keep)
	take = getattr(values, "take", None)
	if take is not None:
		return take(rows, axis=k_axis)
	return np.take(np.asarray(values), rows, axis=k_axis)


def _slab_to_host(a):
	"""Host numpy for one Sigma slab, whether it is replicated or SHARDED.

	The star-spread statistic is measured on ONE omega slice (the caller
	slices before calling, so the transfer is the slice, not the cube).  On
	the ``sigma_omega_layout = sharded`` path at P > 1 that slice is still a
	GLOBALLY sharded ``jax.Array`` whose devices are not all addressable
	from this process, and a bare ``np.asarray`` on it raises
	"Fetching value for `jax.Array` that spans non-addressable ... devices"
	-- which is exactly how the sharded layout died at P=4 the first time
	anyone ran it through the k_irr extraction (the extraction landed
	2026-08-08, the sharded layout 2026-07-28, and the two had never met).

	Same reconstruction ``ppm_windows._to_host_np`` uses, and for the same
	reason; kept local because this module is numpy+h5py by charter and does
	not import jax at module scope.  Replicated / numpy operands take the
	fast path unchanged, so the default layout is byte-for-byte untouched.
	"""
	if getattr(a, "is_fully_addressable", True):
		return np.asarray(a)
	from jax.experimental import multihost_utils
	return np.asarray(multihost_utils.process_allgather(a, tiled=True))


def sigma_star_spread_stats(values, rows_to_keep, compact_irr, sym_idx_k,
                            n_sym_spatial, *, k_axis=0, omega_index=None):
	"""How far this array is from its own star relation.  FOUR numbers.

	Computed on the COMPLETE full-BZ array, before any row is dropped —
	that is the only moment it can be computed at all once the artifact
	stops carrying the star members.

	Four rather than one, because one cannot be read.  A raw-element
	residual on a Σ matrix mixes a physical disagreement together with the
	degenerate-manifold basis choice, and the second can dominate the
	first by thirteen orders (see the block above).  So the raw number is
	reported for continuity with the published one, and beside it two
	quantities the gauge cannot move at all:

	``raw_ev``
	    ``max |A − unfold(select(A))|``.  THE published metric, the one
	    that reads 113.08 eV on ``cohsex_debug``'s Hartree array.  Read it
	    as an upper bound, never as an error.
	``diag_ev``
	    Worst per-band ``max − min`` of the REAL diagonal within a star —
	    ``harness.compare_to_bgw``'s ``_star_spread`` and the quantity
	    ``sc_iteration._KSTAR_SPREAD_TOL`` is set against, so the stamped
	    number is comparable to both.  Gauge-free except inside degenerate
	    manifolds.
	``frobenius_ev``, ``trace_ev``
	    Worst ``max − min`` of ``‖A[k]‖_F`` and of ``|Tr A[k]|`` within a
	    star.  Both are invariant under ``A → U†AU`` for ANY unitary U,
	    which is a superset of the diagonaliser's freedom, so a nonzero
	    value here is proof that the operators genuinely differ and not an
	    artifact of anybody's basis.  They are also free: no eigen- or
	    singular-value decomposition, two reductions over an array already
	    in memory.  On the control they collapse 7.78 → 0.003; on the Σ
	    cubes they do not collapse, which is how one reads that the
	    fixture's residual spread is real.

	A 4-D cube is measured at ``omega_index`` alone (the ω nearest zero,
	chosen by the writer), because the star relation is a per-ω statement
	and measuring all of them would cost a second cube of memory for no
	extra verdict.  The index is stamped alongside so the number is never
	read as an all-ω claim.
	"""
	from ffi import _services
	_services.ensure_on_path()
	import symmetry_maps

	# ONE ω SLICE, TAKEN BEFORE THE HOST TRANSFER.  ``np.asarray`` on the
	# cube itself would pull all (n_omega, nk, nb, nb) of it back from the
	# devices to measure one slice of it — 1.3 GB on the mos2_4x4 deck for
	# a 33 MB answer.  Slicing first keeps the transfer to the slice.
	ndim = len(np.shape(values))
	if ndim == 4:
		if omega_index is None:
			omega_index = np.shape(values)[0] // 2
		M = _slab_to_host(values[int(omega_index)])
	else:
		M = _slab_to_host(values)
		omega_index = -1
	if k_axis != 0:
		M = np.moveaxis(M, k_axis - (1 if ndim == 4 else 0), 0)

	sel = M[np.asarray(rows_to_keep)]
	# ``star_row``: ``sel`` is rows taken out of the FULL-BZ array ``M``,
	# so each row carries a ``sym_idx`` of its own and the predicate is the
	# XOR — the same flavour ``star_select`` produces.  This used to ride
	# the argument's default; the default is gone, because the other branch
	# is wrong here by 183.61 eV on the off-diagonals with the real
	# diagonal exactly intact, which nothing downstream would have seen.
	unfolded = np.asarray(symmetry_maps.star_broadcast(
		sel, np.asarray(compact_irr), np.asarray(sym_idx_k),
		int(n_sym_spatial),
		irr_labels=np.arange(len(rows_to_keep), dtype=np.int32),
		trs_reference="star_row"))
	raw = float(np.abs(M - unfolded).max()) if M.size else 0.0

	diag = frob = trace = 0.0
	compact = np.asarray(compact_irr)
	for lab in range(len(rows_to_keep)):
		mem = np.flatnonzero(compact == lab)
		if mem.size < 2:
			continue
		blk = M[mem]
		if blk.ndim == 3:
			d = np.real(np.diagonal(blk, axis1=1, axis2=2))
			diag = max(diag, float((d.max(0) - d.min(0)).max()))
			t = np.abs(np.trace(blk, axis1=1, axis2=2))
			trace = max(trace, float(t.max() - t.min()))
		else:
			d = np.real(blk)
			diag = max(diag, float((d.max(0) - d.min(0)).max()))
		f = np.linalg.norm(blk.reshape(mem.size, -1), axis=1)
		frob = max(frob, float(f.max() - f.min()))

	return {
		"raw_ev": raw,
		"diag_ev": diag,
		"frobenius_ev": frob,
		"trace_ev": trace,
		"omega_index": int(omega_index),
	}


def extract_and_stamp_k_irr(payload, star, *, omega_ev=None, nk_full=None,
                            print_fn=None):
	"""THE RULED ORDERING, in one implementation.  Measure, then drop.

	``payload`` maps dataset name → array (or ``None``, which passes
	through).  The k axis of each name comes from :data:`SIGMA_K_AXIS`, so
	a name that table does not carry is REFUSED rather than written on a
	guessed axis — an array dropped along the wrong axis is still an array
	of the right dtype and a plausible shape.

	Returns ``(payload, attrs_for, compact_irr, sym_idx_k)``.  ``attrs_for``
	is a callable ``name → dict | None`` carrying the storage stamp and
	that dataset's own four spread numbers.

	WHY THIS IS A FUNCTION AND NOT A BLOCK IN THE WRITER.  Two writers now
	need the ordering: :func:`write_sigma_omega_h5`, which creates the
	file, and :func:`append_qsgw_datasets_h5`, which adds the QSGW cube and
	the QP ladders to a file that already exists.  The ordering is the
	owner's ruling — the arrays arrive COMPLETE on the full BZ, the
	star-spread statistic is measured on them, and only then are rows
	dropped — and a second copy of it is a second place for the measure and
	the drop to swap.  They must not: after the drop each star has exactly
	one member left and every spread arm reads identically zero, which
	``tests/test_sigma_kirr_extraction.py`` demonstrates rather than
	asserts.

	``star`` of ``None`` is the full-BZ write.  The payload comes back
	untouched and ``attrs_for`` returns ``None`` for every name, which is
	the no-attr-means-full back-compat direction: a file that carries no
	stamp is read as full-BZ, so nothing written before this format can be
	reinterpreted by it.
	"""
	unknown = sorted(n for n in payload if n not in SIGMA_K_AXIS)
	if unknown:
		raise ValueError(
			f"extract_and_stamp_k_irr: {unknown} are not in SIGMA_K_AXIS, so "
			f"this function does not know which of their axes is k.  Add the "
			f"dataset there with its k axis; do not let it default to 0.")
	if star is None:
		return dict(payload), (lambda name: None), None, None

	irr_idx_k, sym_idx_k, n_sym_spatial = star
	irr_idx_k = np.asarray(irr_idx_k)
	sym_idx_k = np.asarray(sym_idx_k, dtype=np.int32)
	if nk_full is not None and irr_idx_k.size != int(nk_full):
		raise ValueError(
			f"star tables describe {irr_idx_k.size} full-BZ k but the "
			f"Σ cube has nk={int(nk_full)}; the sweep and the tables are not "
			f"from the same run.")
	rows_to_keep, compact_irr = compact_star_tables(irr_idx_k)

	# EVERY ARRAY MUST ARRIVE ON THE FULL BZ.  An already-extracted array
	# handed in here would be measured against a star relation it no longer
	# has the members for and then dropped a second time; the shapes make
	# that statable, so it is stated.
	for name, arr in payload.items():
		if arr is None:
			continue
		k_axis = SIGMA_K_AXIS[name]
		nk_seen = int(np.shape(arr)[k_axis])
		if nk_seen != irr_idx_k.size:
			raise ValueError(
				f"{name}: axis {k_axis} has {nk_seen} rows but the star "
				f"tables describe {irr_idx_k.size} full-BZ k.  This function "
				f"takes COMPLETE full-BZ arrays — it is what makes the "
				f"spread statistic measurable at all — and "
				f"{len(rows_to_keep)} rows would be an already-extracted one.")

	# MEASURED ON THE COMPLETE ARRAYS, BEFORE ANY ROW IS DROPPED.
	omega_index = (int(np.argmin(np.abs(np.asarray(omega_ev))))
	               if omega_ev is not None else None)
	stamps: dict[str, dict] = {}
	for name, arr in payload.items():
		if arr is None:
			continue
		stats = sigma_star_spread_stats(
			arr, rows_to_keep, compact_irr, sym_idx_k, n_sym_spatial,
			k_axis=SIGMA_K_AXIS[name], omega_index=omega_index)
		stamps[name] = stats
		if print_fn is not None:
			print_fn(
				f"  Σ star spread [{name}]: raw {stats['raw_ev']:.4f} eV, "
				f"diag {stats['diag_ev']:.4f} eV, gauge-blind "
				f"‖·‖_F {stats['frobenius_ev']:.4f} / "
				f"|Tr| {stats['trace_ev']:.4f} eV")
	if print_fn is not None:
		print_fn(
			f"  Σ k axis EXTRACTED to k_irr: {irr_idx_k.size} -> "
			f"{len(rows_to_keep)} rows (selection, not reconstruction).")

	# ...and only now are they dropped.
	out = {
		name: (None if arr is None else star_select_k_irr(
			arr, rows_to_keep, k_axis=SIGMA_K_AXIS[name]))
		for name, arr in payload.items()
	}

	def attrs_for(name):
		a = {
			K_STORAGE_ATTR: K_STORAGE_IBZ,
			K_STORAGE_VERSION_ATTR: K_STORAGE_VERSION,
			N_SYM_SPATIAL_ATTR: int(n_sym_spatial),
			"nk_full": int(compact_irr.size),
		}
		for key, val in stamps.get(name, {}).items():
			a[SPREAD_ATTR_PREFIX + key] = val
		return a

	return out, attrs_for, compact_irr, sym_idx_k


def derive_sigma_total(sigma_c, sigma_sx, hartree):
	"""``(c + sx[None]) + h[None]`` — THE association, in one place.

	Both write paths need this sum and they must produce bit-identical
	bytes, so they call one function rather than each spelling their own
	parenthesisation: the replicated derivation in
	:func:`write_sigma_omega_h5` and the sharded one inside
	``gw.dynamic_sigma.write_sigma_omega``'s jitted ``_ev_tensors``.
	Works on host and device arrays alike — it is three adds.

	WHY THIS IS A NAMED FUNCTION AND NOT AN EXPRESSION.  A float64
	reassociation of a three-term sum moves the result at 1e-16 relative,
	which no physics gate would ever catch, and the two paths sat in
	different files with no gate between them.
	``tests/test_sigma_kirr_extraction.py`` pins the association here,
	where pinning it costs one cell instead of a parallel run.

	(The 4.35e-05 eV that ``cohsex_debug``'s committed cube shows against
	this sum is NOT a reassociation — all three orders reproduce it
	bit-identically, and every array in that file is a widened float32.
	See the block above.)
	"""
	total = sigma_c
	if sigma_sx is not None:
		total = total + sigma_sx[None, ...]
	if hartree is not None:
		total = total + hartree[None, ...]
	return total


def k_irr_rows_for(full_bz_indices, compact_irr, *, what="caller"):
	"""Full-BZ row indices → STORED row indices, or a refusal.

	The one adapter a k_irr-side consumer needs, and the reason it refuses
	rather than remapping blindly.  A consumer that already held full-BZ
	indices naming the wedge rows it wants — ``eqp_bgw``'s
	``kirr_to_kfull`` is the live example — can keep those indices and
	come through here; ``compact_irr[i]`` is the stored row holding k *i*.

	THE REFUSAL IS THE WHOLE POINT.  ``compact_irr[i]`` is defined for
	EVERY full-BZ *i*, including ones whose data is not on disk, and for
	those it returns the star's FIRST member instead.  That substitution —
	handing back a different member of the same star — is exactly the
	operation this file refuses, and on a deck whose quadrature is not
	orbit-closed the two differ by up to the 113 eV in the block above.
	So a request for a row that is not itself a stored row raises, and the
	caller learns which k it asked for instead of receiving a plausible
	wrong matrix.

	In practice the check passes because the rows kept are the
	first-occurrence members and ``kirr_to_kfull`` names first occurrences
	too, but "in practice" is not a contract and this is the twin.
	"""
	compact = np.asarray(compact_irr)
	idx = np.asarray(full_bz_indices, dtype=np.int64)
	if idx.size and (idx.min() < 0 or idx.max() >= compact.size):
		raise ValueError(
			f"{what}: full-BZ index out of range for a {compact.size}-k "
			f"mesh (got min {int(idx.min())}, max {int(idx.max())}).")
	# The stored rows are the first occurrences, recovered from the table
	# itself so this function needs no second source of truth.
	_, first = np.unique(compact, return_index=True)
	stored_rows = set(int(v) for v in first)
	bad = sorted({int(i) for i in idx.ravel()} - stored_rows)
	if bad:
		raise ValueError(
			f"{what}: full-BZ k {bad} are not stored rows of this file.  "
			f"The file keeps one member per star ({len(stored_rows)} of "
			f"{compact.size}), and returning a DIFFERENT member of the same "
			f"star in their place is the substitution this format refuses — "
			f"on a deck whose ISDF quadrature is not orbit-closed those "
			f"members disagree by eV, not by round-off.  Regenerate "
			f"sigma_mnk.h5 from a run whose k_irr set matches this consumer.")
	return compact[idx]


def write_sigma_omega_h5(
	filepath,
	omega_ev,
	sigma_total_kij_ev=None,
	*,
	sigma_c_kij_ev=None,
	sigma_sx_kij_ev=None,
	hartree_kij_ev=None,
	mesh=None,
	star=None,
	omega_reference_ev=None,
	omega_reference_provenance=None,
	sigma_regularization=None,
	eval_energies_rel_ev=None,
	eval_energies_provenance=None,
	omega_coverage=None,
	band_extrapolation=None,
	print_fn=None,
):
	"""Write frequency-dependent Sigma_mnk(omega) arrays to HDF5.

	Datasets:
	  - omega_ev: (n_omega,)                           — rank-0 metadata
	  - sigma_total_kij_ev: (n_omega, nk, nb, nb)      — large sharded
	  - sigma_c_kij_ev  (optional): (n_omega, nk, nb, nb)
	  - sigma_sx_kij_ev (optional): (nk, nb, nb)
	  - hartree_kij_ev  (optional): (nk, nb, nb)
	  - sigma_c_extrap_*_kn_ev (optional): (nk, nb) — the Σ_c
	    band-convergence fit, present only when the run extrapolated

	``band_extrapolation``
	    ``{"arrays": {name: (nk, nb)}, "attrs": {...}}`` from
	    ``gw.band_extrapolation.extrapolation_h5_payload``.  Until
	    2026-08-15 the fitted ``S_inf`` reached NO artifact: a run with the
	    feature on and one with it off were identical to 8e-15 in every
	    dataset here while the log reported an 848 meV correction, so the
	    feature could not be gated, diffed or consumed, and a star-spread
	    test on this file passed vacuously by measuring the
	    un-extrapolated cube.  The arrays ride the SAME extraction and
	    stamp as the cubes rather than being appended raw, so ``S_inf``
	    and ``sigma_c_kij_ev`` cannot disagree about which k the file
	    holds.  ``None`` writes exactly the pre-feature file.

	All large writes go through :mod:`file_io.slab_io`, which has one
	transport (per-rank collective MPI-IO) and no selector.

	``star``
	    ``(irr_idx_k, sym_idx_k, n_sym_spatial)`` — the same triple
	    ``gw.kin_ion_io.star_tables`` returns.  Given it, the k axis of
	    every array above is EXTRACTED to one row per star before it is
	    written, the two unfold tables are filed beside the arrays, and
	    each dataset is stamped so a reader knows what it is holding.
	    Omitted, the full BZ is written exactly as before and no attr
	    appears — which is the back-compat direction that matters, since
	    a dataset with no stamp is read as full-BZ.

	    THE ORDER IS THE RULING, and it is visible in the code below: the
	    arrays arrive complete, the star-spread statistic is measured on
	    the COMPLETE arrays, and only then are rows dropped.  Measuring
	    after the drop would measure nothing — there would be one member
	    per star left to compare.

	``omega_reference_ev`` / ``omega_reference_provenance``
	    THE ENERGY ``omega_ev`` IS MEASURED FROM, and where it came from
	    ("fixed-N mu" or "midgap").  ``omega_ev`` is a RELATIVE axis and
	    always was; until this stamp existed the file did not say relative
	    to WHAT, so every post-hoc consumer had to guess, and
	    ``gw.eqp_bgw.make_eqp_bgw`` guessed the insulating midgap — a
	    measured 2.79 eV mis-sampling of Σ_c(ω) on the sodium metal deck
	    (audit A2).  Optional only so a pre-stamp file still writes; a
	    metallic consumer REFUSES an unstamped file rather than assume.

	``eval_energies_rel_ev`` / ``eval_energies_provenance``
	    THE SPECTRUM THIS Σ WAS EVALUATED AT, on the same relative axis as
	    ``omega_ev``, and which of the two candidates it is
	    (:data:`SIGMA_EVAL_AT_E_DFT` / :data:`SIGMA_EVAL_SELF_CONSISTENT`).
	    A DIFFERENT question from the ω reference, and the one the file did
	    not answer: ``gw.eqp_bgw.make_eqp_bgw``'s own docstring says
	    "Nothing in the files distinguishes the two cases, so this function
	    will not guess", and absent this stamp it then linearizes at E_DFT —
	    right for a one-shot cube, a different and wrong calculation for a
	    self-consistent one.  Optional only so a pre-stamp file still
	    writes; it rides the SAME star extraction as the cubes, so the eval
	    spectrum and the Σ rows cannot disagree about which k they are.

	``omega_coverage``
	    A ``gw.dynamic_sigma.OmegaCoverage`` (duck-typed: ``n_uncovered``,
	    ``fraction_uncovered``, ``policy``).  Stamped on the eval dataset so
	    a consumer centring on those energies learns how many of them the
	    cube cannot actually answer for — the Na semicore run wrote 41.3 %
	    endpoint values as if they were Σ at the state's own energy.
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

	# sigma_total_kij_ev is derived when not passed: total = c + sx + h.
	#
	# THE ASSOCIATION IS ``(c + sx) + h``, and it is the same one
	# ``gw.ppm_pipeline``'s sharded branch spells out for itself, so the
	# two write paths agree bit-for-bit rather than to a tolerance.
	# ``tests/test_sigma_kirr_extraction.py`` pins it.  (The 4.35e-05 eV
	# the cohsex_debug fixture shows against this sum is NOT reassociation
	# — every array in that file is a widened float32 — see the block
	# above.)  This runs BEFORE the extraction so the derived total is the
	# sum of the full-BZ operands and the extraction of the total equals
	# the total of the extraction.
	total = sigma_total_kij_ev
	if total is None:
		total = derive_sigma_total(
			sigma_c_kij_ev, sigma_sx_kij_ev, hartree_kij_ev)

	payload = {
		"sigma_total_kij_ev": total,
		"sigma_c_kij_ev": sigma_c_kij_ev,
		"sigma_sx_kij_ev": sigma_sx_kij_ev,
		"hartree_kij_ev": hartree_kij_ev,
	}
	# INTO THE SAME PAYLOAD as the cubes: the eval spectrum labels the same
	# (k, n) the cube's band diagonal does, so it must be extracted to the
	# same rows by the same table.  Extracting it separately — or not at
	# all — is how a stamp comes to describe rows the file no longer holds.
	if eval_energies_rel_ev is not None:
		payload[SIGMA_EVAL_DATASET] = np.asarray(
			eval_energies_rel_ev, dtype=np.float64)
	extrap_arrays = dict((band_extrapolation or {}).get("arrays", {}))
	extrap_attrs = dict((band_extrapolation or {}).get("attrs", {}))
	# Into the SAME payload, so the extraction, the stamp and the spread
	# measurement are one code path for the cubes and the fit alike.
	payload.update(extrap_arrays)

	# The ordering is :func:`extract_and_stamp_k_irr`'s, shared with the
	# QSGW appender rather than spelled twice.
	payload, _attrs, compact_irr, sym_idx_k = extract_and_stamp_k_irr(
		payload, star, omega_ev=omega_ev, nk_full=nk, print_fn=print_fn)
	if star is not None:
		nk = int(np.shape(payload["sigma_total_kij_ev"])[1])
		shape_ref = (n_omega, nk, nb, nb2)

	eval_rel_extracted = payload.pop(SIGMA_EVAL_DATASET, None)
	total = payload["sigma_total_kij_ev"]
	sigma_c_kij_ev = payload["sigma_c_kij_ev"]
	sigma_sx_kij_ev = payload["sigma_sx_kij_ev"]
	hartree_kij_ev = payload["hartree_kij_ev"]

	with SlabIO(abs_path, mode="w", mesh=mesh) as io:
		io.write_attr("omega_ev", np.asarray(omega_ev, dtype=np.float64))
		# The ω axis's own reference, stamped ON the ω axis — one place to
		# look, and it cannot drift away from the array it describes.
		if omega_reference_ev is not None:
			io.stamp_dataset_attrs(OMEGA_DATASET, {
				OMEGA_REFERENCE_ATTR: float(omega_reference_ev),
				OMEGA_REFERENCE_PROVENANCE_ATTR: str(
					omega_reference_provenance or "unstated"),
			})
		if sigma_regularization is not None:
			# The EFFECTIVE broadening, from the same resolver the kernel
			# ran at (``gw.ppm_windows.resolve_sigma_regularization``) --
			# re-derived from the config here, not threaded, so the two
			# cannot disagree by construction.
			io.stamp_dataset_attrs(OMEGA_DATASET, {
				SIGMA_REGULARIZATION_ATTR:
					float(sigma_regularization.resolved_ev),
				SIGMA_REGULARIZATION_REQUESTED_ATTR:
					float(sigma_regularization.requested_ev),
				SIGMA_REGULARIZATION_FLOOR_POLICY_ATTR:
					str(sigma_regularization.floor_policy),
			})
		if star is not None:
			# The tables live in the same file as the arrays they describe.
			# A table that lives elsewhere is a table that silently decays
			# when anything upstream is regenerated — kin_ion's words, and
			# the same reasoning put them beside the slab there.
			io.write_attr(IRR_IDX_DATASET,
				np.asarray(compact_irr, dtype=np.int32))
			io.write_attr(SYM_IDX_DATASET,
				np.asarray(sym_idx_k, dtype=np.int32))
		io.create_dataset("sigma_total_kij_ev",
			shape=shape_ref, dtype=np.complex128,
			attrs=_attrs("sigma_total_kij_ev"))
		io.write_slab("sigma_total_kij_ev", total)
		if sigma_c_kij_ev is not None:
			io.create_dataset("sigma_c_kij_ev",
				shape=shape_ref, dtype=np.complex128,
				attrs=_attrs("sigma_c_kij_ev"))
			io.write_slab("sigma_c_kij_ev", sigma_c_kij_ev)
		if sigma_sx_kij_ev is not None:
			io.create_dataset("sigma_sx_kij_ev",
				shape=tuple(sigma_sx_kij_ev.shape),
				dtype=np.complex128,
				attrs=_attrs("sigma_sx_kij_ev"))
			io.write_slab("sigma_sx_kij_ev", sigma_sx_kij_ev)
		if hartree_kij_ev is not None:
			io.create_dataset("hartree_kij_ev",
				shape=tuple(hartree_kij_ev.shape),
				dtype=np.complex128,
				attrs=_attrs("hartree_kij_ev"))
			io.write_slab("hartree_kij_ev", hartree_kij_ev)
		if eval_rel_extracted is not None:
			# The two facts that make this array usable ride ON it: which
			# spectrum it is, and how much of it the ω grid could answer
			# for.  A number without either is what the file already had.
			at = dict(_attrs(SIGMA_EVAL_DATASET) or {})
			at[SIGMA_EVAL_PROVENANCE_ATTR] = str(
				eval_energies_provenance or "unstated")
			if omega_coverage is not None:
				at[OMEGA_COVERAGE_N_ATTR] = int(
					getattr(omega_coverage, "n_uncovered", 0))
				at[OMEGA_COVERAGE_FRAC_ATTR] = float(
					getattr(omega_coverage, "fraction_uncovered", 0.0))
				at[OMEGA_COVERAGE_POLICY_ATTR] = str(
					getattr(omega_coverage, "policy", "unstated"))
			io.create_dataset(SIGMA_EVAL_DATASET,
				shape=tuple(np.shape(eval_rel_extracted)),
				dtype=np.float64, attrs=at)
			io.write_slab(SIGMA_EVAL_DATASET,
				np.asarray(eval_rel_extracted, dtype=np.float64))
		for name in extrap_arrays:
			arr = payload[name]
			if arr is None:
				continue
			arr = np.asarray(arr)
			# The run-level facts (band counts, fractions, verdict, any
			# planner fallback) ride on EVERY one of these datasets, so a
			# reader that opens one of them alone still learns whether the
			# number it is holding was trusted.
			at = dict(_attrs(name) or {})
			at.update(extrap_attrs)
			io.create_dataset(name, shape=tuple(arr.shape),
				dtype=np.complex128,
				attrs=at)
			io.write_slab(name, arr)
	return abs_path


def read_omega_reference(filepath):
	"""``(reference_ev, provenance)`` off a ``sigma_mnk.h5``, or ``(None, None)``.

	THE ONE READER OF THE STAMP, so its location is stated once.  ``None``
	means the file predates the stamp (audit A2) — not that its ω axis is
	absolute.  A consumer that cannot tolerate a guess must REFUSE on
	``None`` rather than substitute its own convention; that substitution,
	made silently, is the defect the stamp exists to close.
	"""
	abs_path = os.path.abspath(filepath)
	with h5py.File(abs_path, "r") as h5:
		if OMEGA_DATASET not in h5:
			return None, None
		attrs = h5[OMEGA_DATASET].attrs
		if OMEGA_REFERENCE_ATTR not in attrs:
			return None, None
		ref = float(attrs[OMEGA_REFERENCE_ATTR])
		prov = attrs.get(OMEGA_REFERENCE_PROVENANCE_ATTR, "unstated")
	if isinstance(prov, bytes):
		prov = prov.decode("utf-8")
	return ref, str(prov)


def read_eval_energies(filepath):
	"""``(eval_rel_ev, provenance, coverage)`` off a ``sigma_mnk.h5``.

	``(None, None, None)`` means the file predates the stamp — NOT that the
	cube was evaluated at E_DFT.  A consumer must treat the two differently:
	the second is a fact it can act on, the first is an absence, and
	collapsing them is exactly how ``make_eqp_bgw`` came to linearize a
	self-consistent cube at E_DFT without saying so.

	``coverage`` is ``{"n_uncovered", "fraction_uncovered", "policy"}`` when
	the writer stamped it, else ``None``.

	The array comes back on the file's OWN k rows (the star wedge when the
	file carries one), like every other dataset here; the caller remaps
	through the same ``k_irr_rows_for`` it uses for the cubes.
	"""
	abs_path = os.path.abspath(filepath)
	with h5py.File(abs_path, "r") as h5:
		if SIGMA_EVAL_DATASET not in h5:
			return None, None, None
		ds = h5[SIGMA_EVAL_DATASET]
		arr = np.asarray(ds[()], dtype=np.float64)
		prov = ds.attrs.get(SIGMA_EVAL_PROVENANCE_ATTR, "unstated")
		cov = None
		if OMEGA_COVERAGE_N_ATTR in ds.attrs:
			pol = ds.attrs.get(OMEGA_COVERAGE_POLICY_ATTR, "unstated")
			cov = {
				"n_uncovered": int(ds.attrs[OMEGA_COVERAGE_N_ATTR]),
				"fraction_uncovered": float(
					ds.attrs.get(OMEGA_COVERAGE_FRAC_ATTR, 0.0)),
				"policy": (pol.decode("utf-8")
				           if isinstance(pol, bytes) else str(pol)),
			}
	if isinstance(prov, bytes):
		prov = prov.decode("utf-8")
	return arr, str(prov), cov


# ===========================================================================
# THE OPT-IN PLOTTING PAYLOAD — the QSGW Σ_xc cube and the QP ladders
# ===========================================================================
# These four datasets had NO PRODUCER between 2026-04-11 (``Rewrite
# QP/output section``, which deleted the block that wrote them together
# with a pile of genuinely dead code) and this commit; they survived only
# in ``tests/regression/cohsex_debug/sigma_mnk.h5``, which is why the
# k_irr landing found their names in :data:`SIGMA_K_AXIS` with nothing on
# the other end.  The owner's ruling restores them: gated by the deck,
# default off, "a lot of people will want to plot that".
#
# WHY AN APPEND AND NOT A LONGER ``write_sigma_omega_h5`` SIGNATURE.  On
# the one-shot path the file is written INSIDE
# ``gw.ppm_pipeline.compute_ppm_sigma_pipeline`` (its step 5), and the
# QSGW cube does not exist yet at that moment — ``build_qsgw_sigma_xc``
# runs after the pipeline returns, in ``gw.sigma_dispatch``.  The QP
# ladders are later still: two of the three need ``kin_ion``, which the Σ
# dispatch never sees.  Widening the writer's signature would mean
# holding the whole ω-cube back until the last of its companions existed,
# which is the opposite of what the single-write consolidation bought.
# So each producer writes at its own seam, and they all come through here.
#
# THE FILE, NOT THE CALLER, DECIDES THE k SET.  This function reads its
# own storage from the cube datasets already in it (via
# ``kin_ion.read_star_map``) instead of taking a ``star`` argument.  That
# is not tidiness: a caller-supplied triple that disagreed with the file
# would produce a HETEROGENEOUS artifact — some datasets on the wedge and
# some on the full BZ, all of them shaped plausibly — and no reader in the
# tree checks across datasets.  Reading the file also means every refusal
# ``read_star_map`` already owns (a truncated slab, tables of the wrong
# length, an unknown version, a storage value that is neither legal one)
# guards the append too, with no second copy of any of them here.


def append_qsgw_datasets_h5(filepath, payload, *, print_fn=None):
	"""Add the QSGW / QP-ladder datasets to an existing ``sigma_mnk.h5``.

	Parameters
	----------
	filepath
		An existing ``sigma_mnk.h5``.  This function never CREATES one:
		the datasets describe a Σ that file already holds, and a file with
		only the appendix in it would answer ``sigma_c_kij_ev`` with a
		``KeyError`` where today's consumers get an honest
		``FileNotFoundError``.
	payload
		``name → array`` for any subset of :data:`QSGW_PLOT_DATASETS`.
		Arrays arrive on the FULL BZ whatever the file stores, because the
		star-spread statistic is measured before the rows are dropped and
		it cannot be measured any other way.  A ``None`` value is skipped,
		which is how a mode that did not build a quantity says so.
	print_fn
		Rank-0 print.  Gets the spread lines and one summary naming every
		dataset written.

	Returns the list of dataset names actually written.

	CALL THIS ON RANK 0 ONLY, with a barrier after — h5py is a
	single-writer library and every array here is replicated (the QSGW
	cube is forced replicated by ``qsgw_utils.build_qsgw_sigma_xc``'s
	final ``with_sharding_constraint``; the ladders are host eigenvalues).
	There is no sharded slab to write, so ``SlabIO``'s collective
	transport would buy nothing and would make the write impossible on any
	box without ``liblorrax_ffi``.
	"""
	abs_path = os.path.abspath(filepath)
	if not os.path.isfile(abs_path):
		raise FileNotFoundError(
			f"append_qsgw_datasets_h5: {abs_path} does not exist.  These "
			f"datasets are an APPENDIX to sigma_mnk.h5 and this function "
			f"does not create one; a run whose compute mode writes no Σ_c(ω) "
			f"cube has nowhere to put them.")

	payload = {k: v for k, v in payload.items() if v is not None}
	if not payload:
		return []

	# The file's OWN k storage, learned from a dataset it was created with.
	with h5py.File(abs_path, "r") as h5:
		present = [n for n in SIGMA_CUBE_DATASETS if n in h5]
		omega_ev = (np.asarray(h5["omega_ev"][()])
		            if "omega_ev" in h5 else None)
		nk_stored = (int(h5[present[0]].shape[SIGMA_K_AXIS[present[0]]])
		             if present else 0)
	if not present:
		raise ValueError(
			f"append_qsgw_datasets_h5: {os.path.basename(abs_path)} holds "
			f"none of {list(SIGMA_CUBE_DATASETS)}, so it is not a "
			f"sigma_mnk.h5 and its k storage cannot be read off it.  "
			f"Refusing to append rather than guessing the k axis.")
	ref = present[0]
	star = read_star_map(abs_path, ref, k_axis=SIGMA_K_AXIS[ref])
	nk_full = int(np.asarray(star[0]).size) if star is not None else nk_stored

	# Same ordering as the creating writer, same function: measure on the
	# complete arrays, then drop.
	payload, attrs_for, _, _ = extract_and_stamp_k_irr(
		payload, star, omega_ev=omega_ev, nk_full=nk_full, print_fn=print_fn)
	if star is None:
		for name, arr in payload.items():
			nk_seen = int(np.shape(arr)[SIGMA_K_AXIS[name]])
			if nk_seen != nk_stored:
				raise ValueError(
					f"{name}: axis {SIGMA_K_AXIS[name]} has {nk_seen} rows "
					f"but {os.path.basename(abs_path)} stores {nk_stored} on "
					f"the full BZ ({ref}).  Appending a dataset on a "
					f"different k set would make one file mean two things.")

	written = []
	with h5py.File(abs_path, "a") as h5:
		for name in QSGW_PLOT_DATASETS:
			if name not in payload:
				continue
			arr = np.asarray(payload[name])
			arr = arr.astype(np.complex128 if np.iscomplexobj(arr)
			                 else np.float64)
			# Rewritten, not merged: a rerun in the same directory must
			# leave the stamp and the values describing THIS run.
			if name in h5:
				del h5[name]
			ds = h5.create_dataset(name, data=arr)
			for key, val in (attrs_for(name) or {}).items():
				ds.attrs[key] = val
			written.append(name)
	if print_fn is not None and written:
		print_fn(
			f"  QSGW plot datasets -> {os.path.basename(abs_path)}: "
			+ ", ".join(written)
			+ (f" (k_irr, {nk_full} -> {nk_stored} rows)" if star is not None
			   else f" (full BZ, {nk_stored} rows)"))
	return written

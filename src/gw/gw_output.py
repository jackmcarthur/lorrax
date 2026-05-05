"""GW driver output: banner, summary, and result serialization.

Analogous to QE's ``punch()`` / ``pw_restart_new`` — all format-specific
I/O lives here so the driver reads like a Methods section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from common.units import RYD_TO_EV

# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class GWResults:
    """All quantities produced by a GW calculation, ready for serialization.

    Self-energy arrays are in **Rydberg** (the internal unit).  The writer
    converts to eV when producing human-readable files.

    Static-COHSEX components (``sig_sx``, ``sig_coh``) and bare exchange
    (``sig_x``) are always populated.  When ``use_ppm=True`` the dynamic
    correlation diagonal is in ``sigma_c_diag_at_dft_ry`` and the writer
    emits sigX/sigC columns instead of sigSX/sigCOH.

    Attributes
    ----------
    sig_sx : np.ndarray, (nk, nb, nb)
        Static screened-exchange Σ_SX (Ry).  In PPM mode this is still
        the static COHSEX value, retained for diagnostics/restart.
    sig_coh : np.ndarray, (nk, nb, nb)
        Static Coulomb-hole Σ_COH (Ry).
    sig_h : np.ndarray, (nk, nb, nb)
        Hartree self-energy (Ry).
    sig_x : np.ndarray, (nk, nb, nb)
        Bare exchange Σ_X (Ry).  Used as the "sigX" column in PPM mode
        and as a quality-of-fit check in COHSEX mode.
    E_qp_ry : np.ndarray, (nk, nb)
        Quasiparticle eigenvalues from diagonalisation (Rydberg).
    U_qp : np.ndarray, (nk, nb, nb)
        Quasiparticle eigenvectors  U[k,m,n] = ⟨m_DFT|n_QP⟩.
    E_dft_ry : np.ndarray, (nk, nb)
        DFT reference eigenvalues (Rydberg).
    kin_ion_ry : np.ndarray, (nk, nb, nb)
        H₀ = T + V_ion matrix (Rydberg).
    band_start, band_stop : int
        0-based band window [band_start, band_stop).
    use_ppm : bool
        If True, labels switch from SX/COH to X/C in output files and
        the writer pulls the correlated diagonal from
        ``sigma_c_diag_at_dft_ry``.
    self_consistent : bool
        Whether the self-energy was obtained self-consistently.
    sigma_c_diag_at_dft_ry : np.ndarray or None, (nk, nb)
        Diagonal of Σ_c interpolated at DFT energies (Ry).  Present only
        for PPM non-SC runs; the writer expands it to a band-diagonal
        matrix for the eqp0.dat ``sigC`` column.
    sigma_xc_at_dft_ev : np.ndarray or None, (nk, nb)
        Diagonal Σ_xc interpolated at DFT energies (eV).  Present only
        for G₀W₀-PPM non-self-consistent runs.
    sigma_omega_h5_path : str or None
        Path to the frequency-dependent σ(ω) HDF5 file, if written.
    tensors_filename : str or None
        Path to the ISDF restart file, for the closing status line.
    """

    sig_sx: np.ndarray
    sig_coh: np.ndarray
    sig_h: np.ndarray
    sig_x: np.ndarray
    E_qp_ry: np.ndarray
    U_qp: np.ndarray
    E_dft_ry: np.ndarray
    kin_ion_ry: np.ndarray
    band_start: int
    band_stop: int
    use_ppm: bool = False
    self_consistent: bool = False
    sigma_c_diag_at_dft_ry: np.ndarray | None = None
    sigma_xc_at_dft_ev: np.ndarray | None = None
    # Full ω-grid Σ_c diagonal (PPM modes only) — drives the Z-factor in
    # the BGW eqp1.dat writer.  Shape (n_omega, nk, nb_sigma), eV; ω is
    # relative to the DFT mid-gap E_F.  None in static modes ⇒ Z=1.
    sigma_c_omega_diag_ev: np.ndarray | None = None
    omega_rel_ev: np.ndarray | None = None
    sigma_omega_h5_path: str | None = None
    tensors_filename: str | None = None

    # Bispinor-only Σ_X / Σ_H decomposition.  ``sig_x_charge`` is the (0,0)
    # Lorentz tile alone (= the non-bispinor scalar Σ_X) and ``sig_x_transverse``
    # is the sum of the 9 transverse (i,j ∈ {1,2,3}) tiles.  ``sig_h_charge``
    # / ``sig_h_transverse`` decompose the bispinor Hartree similarly across
    # μ_L=0 vs μ_L>0.  ``None`` outside the bispinor path.
    sig_x_charge: np.ndarray | None = None
    sig_x_transverse: np.ndarray | None = None
    sig_h_charge: np.ndarray | None = None
    sig_h_transverse: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Banner / summary  (QE ``summary()`` pattern)
# ---------------------------------------------------------------------------

def print_banner(
    backend: str,
    n_devices: int,
    grid_x: int,
    grid_y: int,
    n_procs: int,
    device_kind: str,
    print_fn=print,
):
    """Print the calculation header — device mesh, XLA pool, etc.

    Corresponds to the "announce what we are about to do" phase that QE's
    ``summary()`` / ``hp_summary()`` routines handle before any computation.
    """
    print_fn("")
    print_fn("=" * 72)
    print_fn("  COHSEX-JAX: Self-Energy Calculation")
    print_fn("=" * 72)
    print_fn(
        f"  Backend: {backend.upper():<8}  Devices: {n_devices}"
        f"  Mesh: {grid_x}×{grid_y}  Processes: {n_procs}"
    )
    print_fn(f"  Device type: {device_kind}")

    _preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset")
    _mem_frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset")
    print_fn(f"  XLA preallocate: {_preallocate}  mem_fraction: {_mem_frac}")

    try:
        import jax
        _stats = jax.devices()[0].memory_stats()
        if _stats:
            _bl = _stats.get("bytes_limit", 0) / 1e9
            _bu = _stats.get("bytes_in_use", 0) / 1e9
            print_fn(
                f"  XLA pool: limit={_bl:.2f} GB, in_use={_bu:.2f} GB,"
                f" avail={_bl - _bu:.2f} GB"
            )
    except Exception:
        pass

    print_fn("=" * 72)
    print_fn("")


def print_section(title: str, print_fn=print):
    """Print a section divider: ---- TITLE ----"""
    print_fn("")
    print_fn("-" * 72)
    print_fn(f"  {title}")
    print_fn("-" * 72)


def print_system_summary(
    n_rmu: int,
    fft_grid: tuple[int, int, int],
    cell_volume: float,
    print_fn=print,
):
    """Print ISDF basis and grid metadata before the computation begins."""
    print_fn(f"  ISDF basis: {n_rmu} centroids")
    print_fn(
        f"  FFT grid: {fft_grid[0]}×{fft_grid[1]}×{fft_grid[2]}"
        f"   Cell volume: {cell_volume:.2f} a.u.³"
    )
    print_fn("")


# ---------------------------------------------------------------------------
# Result writer  (QE ``punch('all')`` pattern)
# ---------------------------------------------------------------------------

def write_results(
    results: GWResults,
    sigma_diag_file: str,
    eqp0_file: str,
    eqp1_file: str,
    input_dir: str,
    kpoints_crys: np.ndarray,
    kgrid: tuple[int, int, int],
    kpoints_irr_frac: np.ndarray,
    kpoints_reduced: np.ndarray | None = None,
    kirr_to_kfull: np.ndarray | None = None,
    print_fn=print,
    *,
    no_degen_averaging: bool = False,
    degen_avg_tol_ry: float = 1.0e-6,
    eqp_dE_ev: float = 0.5,
):
    """Serialize all GW outputs — the unified ``punch('all')`` gateway.

    Writes (always):

    1. ``sigma_diag.dat``  — LORRAX per-(k,n) Σ-decomposition diagnostic.
    2. ``eqp0.dat``        — BGW-format zeroth-order QP energies.
    3. ``eqp1.dat``        — BGW-format Z-linearized QP energies (Z=1 in
       static COHSEX, BGW central-difference Z in PPM modes).
    4. ``qp_wfn_rotations.h5`` — QP eigenvectors for band-structure interp.

    Conditional (PPM, non-SC):

    5. ``eqp_g0w0.dat``    — explicit Re/Im of (H₀ + Σ_xc(E_DFT)) for
       hand-debugging convergence.

    Parameters
    ----------
    results : GWResults
        Populated results container (self-energy arrays in Rydberg).
    sigma_diag_file, eqp0_file, eqp1_file : str
        Output paths for the three text files.
    input_dir : str
        Base directory for ancillary output files (eqp_g0w0.dat,
        qp_wfn_rotations.h5).
    kpoints_crys : np.ndarray, (nk_full, 3)
        Full-zone k-points in crystal coordinates (for qp_wfn_rotations).
    kgrid : (nkx, nky, nkz)
        k-mesh dimensions.
    kpoints_irr_frac : np.ndarray, (nk_irr, 3)
        IBZ-wedge k-points in fractional coords (used for BGW eqp{0,1}.dat).
    kpoints_reduced, kirr_to_kfull : optional
        Reduced-zone k-point mapping for restart metadata.
    eqp_dE_ev : float
        Central-difference spacing for the Z-factor in eqp1.dat.
    """
    from file_io import (
        write_sigma_to_file,
        write_eqp_g0w0,
        write_qp_rotations_h5,
    )
    from .eqp_bgw import write_eqp_bgw_in_memory

    r2e = RYD_TO_EV

    # ── Per-k Σ-decomposition diagnostic (LORRAX-native) ──────────────────
    # ``sigma_diag.dat`` is a human-eyeball dump of the diagonal Σ pieces
    # per (k, n).  Column labels switch on mode: COHSEX prints
    # sigSX/sigCOH/sigTOT/VH; PPM prints sigX/sigC/sigXC/VH (same array
    # slots, relabelled).  The driver passes the right arrays for each mode.
    if results.use_ppm:
        sx_arr = results.sig_x
        diag_ry = results.sigma_c_diag_at_dft_ry
        corr_arr = np.zeros_like(results.sig_coh)
        if diag_ry is not None:
            nb = diag_ry.shape[1]
            idx = np.arange(nb)
            corr_arr[:, idx, idx] = np.asarray(diag_ry)
    else:
        sx_arr = results.sig_sx
        corr_arr = results.sig_coh

    sx_out    = r2e * sx_arr
    corr_out  = r2e * corr_arr
    sig_h_out = r2e * results.sig_h
    sig_x_out = r2e * results.sig_x  # always populated; needed for eqp{0,1}

    # Bispinor breakdown columns (None outside the bispinor path).
    sig_x_q_out = (r2e * results.sig_x_charge) if results.sig_x_charge is not None else None
    sig_x_T_out = (r2e * results.sig_x_transverse) if results.sig_x_transverse is not None else None
    sig_h_q_out = (r2e * results.sig_h_charge) if results.sig_h_charge is not None else None
    sig_h_T_out = (r2e * results.sig_h_transverse) if results.sig_h_transverse is not None else None

    # BGW-style degenerate-set averaging: replace the diagonal of each
    # Σ matrix with the mean over each contiguous degenerate group of
    # DFT eigenvalues.  Mirrors Sigma/shiftenergy.f90 (lines 86-122).
    # Off-diagonal entries are preserved.
    if not no_degen_averaging:
        from .degen_average import apply_to_matrix_diagonals
        e_kn_ry = np.asarray(results.E_dft_ry, dtype=np.float64)
        sx_out    = apply_to_matrix_diagonals(sx_out,    e_kn_ry, degen_avg_tol_ry)
        corr_out  = apply_to_matrix_diagonals(corr_out,  e_kn_ry, degen_avg_tol_ry)
        sig_h_out = apply_to_matrix_diagonals(sig_h_out, e_kn_ry, degen_avg_tol_ry)
        sig_x_out = apply_to_matrix_diagonals(sig_x_out, e_kn_ry, degen_avg_tol_ry)
        if sig_x_q_out is not None:
            sig_x_q_out = apply_to_matrix_diagonals(sig_x_q_out, e_kn_ry, degen_avg_tol_ry)
        if sig_x_T_out is not None:
            sig_x_T_out = apply_to_matrix_diagonals(sig_x_T_out, e_kn_ry, degen_avg_tol_ry)
        if sig_h_q_out is not None:
            sig_h_q_out = apply_to_matrix_diagonals(sig_h_q_out, e_kn_ry, degen_avg_tol_ry)
        if sig_h_T_out is not None:
            sig_h_T_out = apply_to_matrix_diagonals(sig_h_T_out, e_kn_ry, degen_avg_tol_ry)

    write_sigma_to_file(
        sx_out,
        sigma_diag_file,
        sigma_coh_kij_eV=corr_out,
        hartree_kij_eV=sig_h_out,
        sx_label="sigX" if results.use_ppm else "sigSX",
        corr_label="sigC" if results.use_ppm else "sigCOH",
        total_label="sigXC" if results.use_ppm else "sigTOT",
        sigma_x_charge_kij_eV=sig_x_q_out,
        sigma_x_transverse_kij_eV=sig_x_T_out,
        hartree_charge_kij_eV=sig_h_q_out,
        hartree_transverse_kij_eV=sig_h_T_out,
    )

    # ── BGW-format eqp0.dat / eqp1.dat ────────────────────────────────────
    # Single source of truth for the linearization math: ``eqp_bgw``
    # provides the central-difference Z-factor and the Newton update.
    # Static modes (COHSEX) hand in ``sigma_c_omega_diag_ev=None`` ⇒ Z=1
    # ⇒ eqp1 == eqp0, matching BGW's behavior for static runs.
    #
    # All gw_jax internal arrays live on the unfolded full BZ (nk_full);
    # BGW's eqp{0,1}.dat lists only the IBZ wedge.  ``kirr_to_kfull[i]``
    # is the full-BZ index of IBZ point ``i``; index with it to subset.
    if kirr_to_kfull is None:
        raise ValueError(
            "write_results requires kirr_to_kfull for the BGW eqp{0,1}.dat "
            "writer — pass sym.kirr_fullids."
        )
    irr_idx = np.asarray(kirr_to_kfull, dtype=np.int64)

    e_dft_ev_full = np.asarray(results.E_dft_ry, dtype=np.float64) * r2e
    e_dft_ev_irr = e_dft_ev_full[irr_idx]
    kin_ion_diag_ev = (
        np.real(np.diagonal(results.kin_ion_ry[irr_idx], axis1=1, axis2=2)) * r2e
    )
    hartree_diag_ev = np.real(np.diagonal(sig_h_out[irr_idx], axis1=1, axis2=2))
    sigma_x_diag_ev = np.real(np.diagonal(sig_x_out[irr_idx], axis1=1, axis2=2))
    # Σ_c at E_DFT diagonal: in PPM mode this is the interpolated value
    # the driver already computed; in static modes it is the static Σ_COH
    # diagonal (post-degen-averaging if enabled).
    if results.use_ppm and results.sigma_c_diag_at_dft_ry is not None:
        sigma_c_at_dft_diag_ev = (
            np.asarray(results.sigma_c_diag_at_dft_ry, dtype=np.complex128)[irr_idx] * r2e
        )
    else:
        sigma_c_at_dft_diag_ev = np.diagonal(corr_out[irr_idx], axis1=1, axis2=2)

    # E_DFT relative to mid-gap E_F (matches gw_jax convention).  Only
    # needed when there is a finite ω-grid to interpolate against.
    sigma_c_omega_diag_ev_irr = None
    e_dft_rel_ev_irr = None
    if results.sigma_c_omega_diag_ev is not None and results.omega_rel_ev is not None:
        # ``sigma_c_omega_diag_ev`` is (n_omega, nk_full, nb).  Subset.
        sigma_c_omega_diag_ev_irr = np.asarray(
            results.sigma_c_omega_diag_ev, dtype=np.complex128
        )[:, irr_idx, :]
        n_occ = int(min(e_dft_ev_full.shape[1], results.band_stop - results.band_start))
        vbm_ev = float(np.max(e_dft_ev_irr[:, :n_occ]))
        cbm_ev = (
            float(np.min(e_dft_ev_irr[:, n_occ:]))
            if n_occ < e_dft_ev_irr.shape[1] else vbm_ev
        )
        efermi_ev = 0.5 * (vbm_ev + cbm_ev)
        e_dft_rel_ev_irr = e_dft_ev_irr - efermi_ev

    write_eqp_bgw_in_memory(
        eqp0_path=eqp0_file, eqp1_path=eqp1_file,
        kpoints_irr_frac=np.asarray(kpoints_irr_frac, dtype=np.float64),
        band_offset=results.band_start,
        e_dft_ev=e_dft_ev_irr,
        kin_ion_diag_ev=kin_ion_diag_ev,
        hartree_diag_ev=hartree_diag_ev,
        sigma_x_diag_ev=sigma_x_diag_ev,
        sigma_c_at_dft_diag_ev=sigma_c_at_dft_diag_ev,
        sigma_c_omega_diag_ev=sigma_c_omega_diag_ev_irr,
        omega_rel_ev=results.omega_rel_ev,
        e_dft_rel_ev=e_dft_rel_ev_irr,
        dE_ev=eqp_dE_ev,
        nspin=1,
    )

    # ── eqp_g0w0.dat (PPM non-SC only) — explicit Re/Im of H₀+Σ_xc(E_DFT) ──
    if (
        not results.self_consistent
        and results.sigma_xc_at_dft_ev is not None
    ):
        h0_diag = (
            np.real(
                np.diagonal(results.kin_ion_ry + results.sig_h, axis1=1, axis2=2)
            )
            * r2e
        )
        g0w0_path = os.path.join(input_dir, "eqp_g0w0.dat")
        write_eqp_g0w0(
            g0w0_path,
            results.E_dft_ry * r2e,
            h0_diag + results.sigma_xc_at_dft_ev,
        )
        print_fn(f"  G0W0 diag (E_DFT):     {g0w0_path}")

    # ── qp_wfn_rotations.h5 — QP eigenvectors ─────────────────────────────
    nkx, nky, nkz = kgrid
    write_qp_rotations_h5(
        os.path.join(input_dir, "qp_wfn_rotations.h5"),
        U_mnk=results.U_qp,
        E_qp_nk=results.E_qp_ry / 2.0,  # Ry → Hartree
        band_start=results.band_start,
        band_stop=results.band_stop,
        kpoints_crys=kpoints_crys,
        nkx=nkx, nky=nky, nkz=nkz,
        kpoints_reduced=kpoints_reduced,
        kirr_to_kfull=kirr_to_kfull,
    )

    # ── Status summary ────────────────────────────────────────────────────
    print_fn(f"\n  Sigma diag:   {sigma_diag_file}")
    print_fn(f"  BGW eqp0:     {eqp0_file}")
    print_fn(f"  BGW eqp1:     {eqp1_file}")
    if results.sigma_omega_h5_path:
        print_fn(f"  Sigma(ω):     {results.sigma_omega_h5_path}")
    if results.tensors_filename:
        print_fn(f"  Restart:      {results.tensors_filename}")
    print_fn("")

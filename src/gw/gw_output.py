"""GW driver output: banner, summary, and result serialization.

Analogous to QE's ``punch()`` / ``pw_restart_new`` — all format-specific
I/O lives here so the driver reads like a Methods section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class GWResults:
    """All quantities produced by a GW calculation, ready for serialization.

    Self-energy arrays are in **Rydberg** (the internal unit).  The writer
    converts to eV when producing human-readable files.

    Attributes
    ----------
    sig_sx : np.ndarray, (nk, nb, nb)
        Self-exchange (COHSEX) or bare exchange (PPM) Σ_SX / Σ_X.
    sig_coh : np.ndarray, (nk, nb, nb)
        Screened-COH self-energy (static COHSEX path).
    sig_h : np.ndarray, (nk, nb, nb)
        Hartree self-energy.
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
        If True, labels switch from SX/COH to X/C in output files.
    self_consistent : bool
        Whether the self-energy was obtained self-consistently.
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
    E_qp_ry: np.ndarray
    U_qp: np.ndarray
    E_dft_ry: np.ndarray
    kin_ion_ry: np.ndarray
    band_start: int
    band_stop: int
    use_ppm: bool = False
    self_consistent: bool = False
    sigma_xc_at_dft_ev: np.ndarray | None = None
    sigma_omega_h5_path: str | None = None
    tensors_filename: str | None = None


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

_RYD2EV = 13.6056980659


def write_results(
    results: GWResults,
    output_file: str,
    input_dir: str,
    kpoints_crys: np.ndarray,
    kgrid: tuple[int, int, int],
    kpoints_reduced: np.ndarray | None = None,
    kirr_to_kfull: np.ndarray | None = None,
    print_fn=print,
    *,
    no_degen_averaging: bool = False,
    degen_avg_tol_ry: float = 1.0e-6,
):
    """Serialize all GW outputs — the unified ``punch('all')`` gateway.

    Parameters
    ----------
    results : GWResults
        Populated results container (self-energy in Rydberg).
    output_file : str
        Path for the main ``eqp0.dat`` file.
    input_dir : str
        Base directory for ancillary output files.
    kpoints_crys : np.ndarray, (nk, 3)
        Full-zone k-points in crystal coordinates.
    kgrid : (nkx, nky, nkz)
        k-mesh dimensions.
    kpoints_reduced, kirr_to_kfull : optional
        Reduced-zone k-point mapping for restart metadata.
    print_fn : callable
        Rank-gated print function.
    """
    from file_io import (
        write_sigma_to_file,
        write_eqp_g0w0,
        write_qp_rotations_h5,
    )

    r2e = _RYD2EV

    # BGW-style degenerate-set averaging: replace the diagonal of each
    # Σ matrix with the mean over each contiguous degenerate group of
    # DFT eigenvalues.  Mirrors Sigma/shiftenergy.f90 (lines 86-122).
    # Off-diagonal entries are preserved.
    sig_sx_out  = r2e * results.sig_sx
    sig_coh_out = r2e * results.sig_coh
    sig_h_out   = r2e * results.sig_h
    if not no_degen_averaging:
        from .degen_average import apply_to_matrix_diagonals
        e_kn_ry = np.asarray(results.E_dft_ry, dtype=np.float64)
        sig_sx_out  = apply_to_matrix_diagonals(sig_sx_out,  e_kn_ry, degen_avg_tol_ry)
        sig_coh_out = apply_to_matrix_diagonals(sig_coh_out, e_kn_ry, degen_avg_tol_ry)
        sig_h_out   = apply_to_matrix_diagonals(sig_h_out,   e_kn_ry, degen_avg_tol_ry)

    # 1. eqp0.dat — main QP self-energy output
    write_sigma_to_file(
        sig_sx_out,
        output_file,
        sigma_coh_kij_eV=sig_coh_out,
        hartree_kij_eV=sig_h_out,
        sx_label="sigX" if results.use_ppm else "sigSX",
        corr_label="sigC" if results.use_ppm else "sigCOH",
        total_label="sigXC" if results.use_ppm else "sigTOT",
    )

    # 2. eqp_g0w0.dat — G₀W₀ diagonal (PPM, non-SC only)
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

    # 3. qp_wfn_rotations.h5 — QP eigenvectors for band-structure interpolation
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

    # 4. Status summary
    print_fn(f"\n  Output: {output_file}")
    if results.sigma_omega_h5_path:
        print_fn(f"  Sigma(ω): {results.sigma_omega_h5_path}")
    if results.tensors_filename:
        print_fn(f"  Restart:  {results.tensors_filename}")
    print_fn("")

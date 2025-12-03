"""Diagnostic script for Hartree calculation in COHSEX.

Run this after a calculation to analyze the Hartree matrix elements
and compare with expected values.
"""

import numpy as np
import jax.numpy as jnp


def diagnose_hartree(
    psi_l_rmu,      # (nk, nb_val, nspinor, n_rmu) - valence wavefunctions at centroids
    psi_r_rmu,      # (nk, nb_sig, nspinor, n_rmu) - sigma wavefunctions at centroids
    V0_munu,        # (n_rmu, n_rmu) - V at q=0 with G=0 excluded
    hartree_kij,    # (nk, nb, nb) - computed Hartree matrix elements
    meta,           # Meta object with system info
    wfn,            # WFNReader with cell info
    print_fn=print,
):
    """Print diagnostic information for Hartree calculation."""
    
    print_fn("\n" + "="*70)
    print_fn("HARTREE DIAGNOSTIC ANALYSIS")
    print_fn("="*70)
    
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    nspin = meta.nspin
    n_rmu = meta.n_rmu
    fft_grid = meta.fft_grid
    n_rtot = np.prod(fft_grid)
    cell_volume = float(wfn.cell_volume)
    
    # System parameters
    print_fn(f"\n1. SYSTEM PARAMETERS:")
    print_fn(f"   k-grid: {meta.kgrid}  (Nk_tot = {nk_tot})")
    print_fn(f"   FFT grid: {fft_grid}  (N_rtot = {n_rtot})")
    print_fn(f"   Cell volume: {cell_volume:.4f} Bohr³")
    print_fn(f"   nspinor: {nspinor}, nspin: {nspin}")
    print_fn(f"   n_rmu (interpolation points): {n_rmu}")
    
    # Wavefunction normalization check
    psi_l = np.asarray(psi_l_rmu)
    psi_r = np.asarray(psi_r_rmu)
    
    psi_norm_l = np.sum(np.abs(psi_l)**2, axis=(0,2,3)) / nk_tot  # sum over k,s,mu, avg over k
    psi_norm_r = np.sum(np.abs(psi_r)**2, axis=(0,2,3)) / nk_tot
    
    print_fn(f"\n2. WAVEFUNCTION NORMS AT CENTROIDS (should be ~O(n_rmu)):")
    print_fn(f"   psi_l |ψ|²: min={psi_norm_l.min():.2f}, max={psi_norm_l.max():.2f}, mean={psi_norm_l.mean():.2f}")
    print_fn(f"   psi_r |ψ|²: min={psi_norm_r.min():.2f}, max={psi_norm_r.max():.2f}, mean={psi_norm_r.mean():.2f}")
    
    # Density at centroids
    rho_mu = np.einsum('knsx,knsx->x', np.conj(psi_l), psi_l).real / nk_tot
    total_density = np.sum(rho_mu)
    
    print_fn(f"\n3. DENSITY AT CENTROIDS:")
    print_fn(f"   ρ_μ: min={rho_mu.min():.6f}, max={rho_mu.max():.6f}, sum={total_density:.4f}")
    print_fn(f"   Number of valence bands: {psi_l.shape[1]}")
    print_fn(f"   Expected ~n_val × n_rmu / N_rtot: {psi_l.shape[1] * n_rmu / n_rtot:.4f}")
    
    # V0 matrix analysis
    V0 = np.asarray(V0_munu)
    V0_diag = np.diag(V0).real
    V0_trace = np.trace(V0).real
    V0_eigenvalues = np.linalg.eigvalsh(V0).real
    
    print_fn(f"\n4. V0 MATRIX (q=0, G≠0 Coulomb in ISDF basis):")
    print_fn(f"   Shape: {V0.shape}")
    print_fn(f"   Diagonal: min={V0_diag.min():.4f}, max={V0_diag.max():.4f}")
    print_fn(f"   Trace: {V0_trace:.4f}")
    print_fn(f"   Eigenvalues: min={V0_eigenvalues.min():.4f}, max={V0_eigenvalues.max():.4f}")
    print_fn(f"   |V0|_Frobenius: {np.linalg.norm(V0):.4f}")
    print_fn(f"   Is Hermitian: {np.allclose(V0, V0.conj().T, atol=1e-10)}")
    
    # Vrho = V0 @ rho
    Vrho = V0 @ rho_mu
    print_fn(f"\n5. HARTREE POTENTIAL AT CENTROIDS (V_H = V0 @ ρ):")
    print_fn(f"   V_H(μ): min={Vrho.real.min():.4f}, max={Vrho.real.max():.4f}, mean={Vrho.real.mean():.4f}")
    print_fn(f"   |Im(V_H)|_max: {np.abs(Vrho.imag).max():.2e} (should be ~0)")
    
    # Hartree matrix elements
    hartree_arr = np.asarray(hartree_kij)
    hartree_diag = np.diagonal(hartree_arr, axis1=-2, axis2=-1).real
    
    print_fn(f"\n6. HARTREE MATRIX ELEMENTS ⟨n|V_H|n⟩ (Rydberg):")
    print_fn(f"   k=0 diagonal elements (first 10 bands):")
    for n in range(min(10, hartree_diag.shape[1])):
        print_fn(f"      n={n}: {hartree_diag[0, n]:.6f} Ry = {hartree_diag[0, n]*13.6:.4f} eV")
    
    # Check for self-consistency: ⟨n|V_H|n⟩ should be positive for bound states
    neg_diag = np.sum(hartree_diag < 0)
    print_fn(f"\n7. SANITY CHECKS:")
    print_fn(f"   Negative diagonal elements: {neg_diag} out of {hartree_diag.size}")
    print_fn(f"   Max |Im(hartree)|: {np.abs(hartree_arr.imag).max():.2e}")
    
    # Expected scale for Hartree
    # V_H ~ 4π × n_el / (a_0 × k²) where k² ~ 1/Ω^{2/3}
    # Very rough estimate: V_H ~ 4π × n_el × Ω^{1/3}
    n_el_est = psi_l.shape[1] * 2 / nspin  # rough electron count
    V_H_scale_est = 4 * np.pi * n_el_est / (cell_volume ** (1/3))
    
    print_fn(f"\n8. SCALE ESTIMATES:")
    print_fn(f"   Estimated electron count: ~{n_el_est:.0f}")
    print_fn(f"   Very rough V_H scale (4πn/Ω^{{1/3}}): {V_H_scale_est:.2f} Ry")
    print_fn(f"   Actual k=0 band 0 V_H: {hartree_diag[0, 0]:.2f} Ry")
    
    # Normalization factor analysis
    print_fn(f"\n9. POTENTIAL NORMALIZATION ISSUES:")
    print_fn(f"   If spin factor missing: code/true = 0.5 (code too small)")
    print_fn(f"   If FFT normalization wrong by √N: factor = √{n_rtot:.0f} = {np.sqrt(n_rtot):.1f}")
    print_fn(f"   If volume factor wrong: Ω = {cell_volume:.1f}")
    print_fn(f"   n_rtot / cell_volume = {n_rtot / cell_volume:.2f}")
    
    print_fn("="*70 + "\n")
    
    return {
        'rho_mu': rho_mu,
        'V0': V0,
        'Vrho': Vrho,
        'hartree_diag': hartree_diag,
        'psi_norm_l': psi_norm_l,
        'psi_norm_r': psi_norm_r,
    }


def compare_with_qe(hartree_kij, qe_V_H_Ry, bands_to_compare=None, print_fn=print):
    """Compare computed Hartree with QE reference values.
    
    Args:
        hartree_kij: (nk, nb, nb) computed Hartree in Rydberg
        qe_V_H_Ry: array of QE V_H values in Ry, shape (nk, nb) or (nb,)
        bands_to_compare: list of band indices to compare (0-indexed)
    """
    hartree_arr = np.asarray(hartree_kij)
    hartree_diag = np.diagonal(hartree_arr, axis1=-2, axis2=-1).real  # (nk, nb)
    
    qe_arr = np.atleast_2d(np.asarray(qe_V_H_Ry))
    if qe_arr.shape[0] == 1:
        qe_arr = np.tile(qe_arr, (hartree_diag.shape[0], 1))
    
    if bands_to_compare is None:
        bands_to_compare = list(range(min(hartree_diag.shape[1], qe_arr.shape[1])))
    
    print_fn("\nCOMPARISON WITH QE (k=0):")
    print_fn(f"{'Band':>6} {'Code (Ry)':>12} {'QE (Ry)':>12} {'Ratio':>10} {'Diff (Ry)':>12}")
    print_fn("-" * 56)
    
    ratios = []
    for n in bands_to_compare:
        if n < hartree_diag.shape[1] and n < qe_arr.shape[1]:
            code_val = hartree_diag[0, n]
            qe_val = qe_arr[0, n]
            ratio = code_val / qe_val if abs(qe_val) > 1e-10 else float('nan')
            diff = code_val - qe_val
            ratios.append(ratio)
            print_fn(f"{n:>6} {code_val:>12.4f} {qe_val:>12.4f} {ratio:>10.4f} {diff:>12.4f}")
    
    ratios = np.array(ratios)
    print_fn("-" * 56)
    print_fn(f"Mean ratio: {np.nanmean(ratios):.4f}")
    print_fn(f"Ratio range: [{np.nanmin(ratios):.4f}, {np.nanmax(ratios):.4f}]")


# Example usage:
if __name__ == "__main__":
    print("This module provides diagnostic functions for Hartree analysis.")
    print("Import and call diagnose_hartree() after a COHSEX calculation.")


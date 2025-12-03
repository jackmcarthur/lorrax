"""
Compare Hartree from ISDF representation vs standard DFT Poisson solver.

This helps identify where the normalization discrepancy arises.

KEY INSIGHT: The ISDF Hartree values are ~20-30x too large compared to QE reference.
This diagnostic helps identify the source of the normalization mismatch between:
1. get_DFT_mtxels.py (correct, uses ortho FFT + explicit volume factors)
2. cohsex_jax.py (incorrect, uses unscaled FFT + sqrt(n_rtot) wfn scaling)
"""

import numpy as np
import jax.numpy as jnp


def compute_hartree_standard(
    psi_G_k,      # (nk, nb_occ, nspinor, nx, ny, nz) - G-space wavefunctions
    bdot,         # (3, 3) reciprocal metric
    cell_volume,  # float
    nk_tot,       # int
):
    """Compute V_H(r) using standard Poisson solver, return diagonal matrix elements.
    
    This matches the approach in get_DFT_mtxels.py.
    Uses norm='ortho' FFTs and proper volume scaling.
    """
    nk, nb_occ, nspinor, nx, ny, nz = psi_G_k.shape
    ngrid = nx * ny * nz
    scale = jnp.sqrt(ngrid / cell_volume)
    
    # Build valence density on real-space grid
    rho_r = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    for ik in range(nk):
        psi_r = jnp.fft.ifftn(psi_G_k[ik], axes=(-3, -2, -1), norm='ortho') * scale
        rho_r = rho_r + jnp.sum(jnp.real(jnp.conj(psi_r) * psi_r), axis=(0, 1))
    rho_r = rho_r / nk_tot
    
    # Poisson solver: V_H(G) = 8π ρ(G) / G² with G=0 = 0
    rho_G = jnp.fft.fftn(rho_r, norm='ortho')
    
    fx = jnp.fft.fftfreq(nx) * nx
    fy = jnp.fft.fftfreq(ny) * ny
    fz = jnp.fft.fftfreq(nz) * nz
    
    M = jnp.asarray(bdot, dtype=jnp.float64)
    ix, iy, iz = fx[:, None, None], fy[None, :, None], fz[None, None, :]
    G2 = (M[0, 0] * ix * ix + M[1, 1] * iy * iy + M[2, 2] * iz * iz
          + 2.0 * M[0, 1] * ix * iy + 2.0 * M[0, 2] * ix * iz + 2.0 * M[1, 2] * iy * iz)
    
    zero_mask = (jnp.arange(nx)[:, None, None] == 0) & \
                (jnp.arange(ny)[None, :, None] == 0) & \
                (jnp.arange(nz)[None, None, :] == 0)
    G2_safe = jnp.where(zero_mask, 1.0, G2)
    
    V_H_G = (8.0 * jnp.pi) * (rho_G / G2_safe)
    V_H_G = V_H_G.at[0, 0, 0].set(0.0)
    V_H_r = jnp.real(jnp.fft.ifftn(V_H_G, norm='ortho'))
    
    return rho_r, V_H_r


def compute_hartree_mtxel_standard(psi_G_k, V_H_r, cell_volume, Gk_crys_all):
    """Compute ⟨n|V_H|n⟩ using standard real-space multiplication.
    
    Returns diagonal elements for k=0.
    """
    nk, nb, nspinor, nx, ny, nz = psi_G_k.shape
    ngrid = nx * ny * nz
    scale = jnp.sqrt(ngrid / cell_volume)
    deltaV = cell_volume / ngrid
    fft_norm = jnp.sqrt(ngrid)
    
    # k=0 only for comparison
    ik = 0
    psi_G = psi_G_k[ik]  # (nb, nspinor, nx, ny, nz)
    Gk = Gk_crys_all[ik]  # (nG, 3)
    Gx = Gk[:, 0].astype(jnp.int32)
    Gy = Gk[:, 1].astype(jnp.int32)
    Gz = Gk[:, 2].astype(jnp.int32)
    
    psi_r = jnp.fft.ifftn(psi_G, axes=(-3, -2, -1), norm='ortho') * scale
    phi_r = psi_r * V_H_r
    phi_G = jnp.fft.fftn(phi_r, axes=(-3, -2, -1), norm='ortho') * (deltaV * fft_norm)
    
    psi_coeffs = psi_G[:, :, Gx, Gy, Gz]
    vpsi = phi_G[:, :, Gx, Gy, Gz]
    
    # Diagonal elements ⟨n|V_H|n⟩
    V_H_diag = jnp.einsum('nsg,nsg->n', jnp.conj(psi_coeffs), vpsi).real
    V_H_diag = V_H_diag * jnp.sqrt(1.0 / cell_volume)
    
    return V_H_diag


def compute_hartree_isdf(
    psi_rmu,      # (nk, nb_occ, nspinor, n_rmu) - wfns at interpolation points  
    psi_rmu_sig,  # (nk, nb_sig, nspinor, n_rmu) - sigma window wfns at interpolation pts
    V0_munu,      # (n_rmu, n_rmu) - ISDF Coulomb at q=0
    nk_tot,       # int
):
    """Compute Hartree matrix elements using ISDF representation.
    
    Returns diagonal elements for k=0.
    """
    # Density at interpolation points
    rho_mu = jnp.einsum('knsx,knsx->x', jnp.conj(psi_rmu), psi_rmu, optimize=True).real
    rho_mu = rho_mu / nk_tot
    
    # V_H in ISDF basis
    Vrho_mu = jnp.einsum('xy,y->x', V0_munu, rho_mu, optimize=True)
    
    # Diagonal matrix elements at k=0
    psi_k0 = psi_rmu_sig[0]  # (nb, nspinor, n_rmu)
    V_H_diag = jnp.einsum('nsx,x,nsx->n', jnp.conj(psi_k0), Vrho_mu, psi_k0, optimize=True).real
    
    return rho_mu, Vrho_mu, V_H_diag


def diagnose_normalization(
    psi_rmu,        # (nk, nb, nspinor, n_rmu) - raw wavefunctions at centroids
    V0_munu,        # (n_rmu, n_rmu) - ISDF Coulomb matrix
    cell_volume,    # float
    n_rtot,         # int - total FFT grid points
    nk_tot,         # int
    n_electrons,    # int - number of valence electrons
    print_fn=print,
):
    """Diagnose normalization issues in the ISDF Hartree calculation.
    
    The key checks are:
    1. Wavefunction normalization at centroids
    2. Density normalization (should integrate to n_electrons)
    3. V_munu magnitude and structure
    4. Expected Hartree matrix element scale
    """
    print_fn("\n" + "="*70)
    print_fn("ISDF HARTREE NORMALIZATION DIAGNOSTIC")
    print_fn("="*70)
    
    psi = np.asarray(psi_rmu)
    V0 = np.asarray(V0_munu)
    nk, nb, ns, n_rmu = psi.shape
    
    print_fn(f"\n1. SYSTEM PARAMETERS:")
    print_fn(f"   n_rtot (FFT grid points) = {n_rtot}")
    print_fn(f"   n_rmu (interpolation pts) = {n_rmu}")
    print_fn(f"   cell_volume = {cell_volume:.4f} Bohr³")
    print_fn(f"   nk_tot = {nk_tot}")
    print_fn(f"   n_electrons = {n_electrons}")
    print_fn(f"   n_bands (valence) = {nb}")
    
    # 2. Check wavefunction normalization
    # If psi was scaled by sqrt(n_rtot), then |psi|² at a point ~ n_rtot
    psi_sq = np.abs(psi)**2
    avg_psi_sq = np.mean(psi_sq)
    
    print_fn(f"\n2. WAVEFUNCTION NORMALIZATION AT CENTROIDS:")
    print_fn(f"   Mean |ψ(r_μ)|² = {avg_psi_sq:.4f}")
    print_fn(f"   sqrt(n_rtot) = {np.sqrt(n_rtot):.2f}")
    print_fn(f"   n_rtot = {n_rtot}")
    print_fn(f"   Expected for unit-normalized ψ: |ψ|² ~ 1/Ω = {1/cell_volume:.6f}")
    print_fn(f"   If scaled by sqrt(n_rtot): |ψ|² ~ n_rtot/Ω = {n_rtot/cell_volume:.2f}")
    
    # 3. Check density normalization
    # rho_mu = (1/Nk) sum_kns |psi_kns(r_mu)|²
    rho_mu = np.einsum('knsx,knsx->x', np.conj(psi), psi).real / nk_tot
    
    print_fn(f"\n3. DENSITY AT CENTROIDS:")
    print_fn(f"   Sum(ρ_μ) = {np.sum(rho_mu):.4f}")
    print_fn(f"   Mean(ρ_μ) = {np.mean(rho_mu):.4f}")
    print_fn(f"   For proper normalization, Σ_μ ρ_μ × ζ_μ(G=0) ~ n_electrons")
    print_fn(f"   But without ζ(G=0): Sum(ρ_μ) ~ n_electrons × n_rmu/n_rtot × (normalization factor)")
    
    # 4. Check V_munu structure
    V0_diag = np.diag(V0).real
    V0_trace = np.trace(V0).real
    V0_eig = np.linalg.eigvalsh(V0).real
    
    print_fn(f"\n4. V_munu STRUCTURE:")
    print_fn(f"   Trace(V0) = {V0_trace:.4f}")
    print_fn(f"   Mean diag = {np.mean(V0_diag):.4f}")
    print_fn(f"   Max eigenvalue = {V0_eig.max():.4f}")
    print_fn(f"   Min eigenvalue = {V0_eig.min():.4f}")
    print_fn(f"   ||V0||_F = {np.linalg.norm(V0):.4f}")
    
    # 5. Estimate expected Hartree scale
    # Physical V_H diagonal ~ 10-30 Ry typically
    Vrho = V0 @ rho_mu
    V_H_diag_est = []
    for n in range(min(nb, 5)):
        psi_n = psi[0, n]  # k=0, band n
        overlap = np.sum(np.abs(psi_n)**2, axis=0)  # sum over spinor
        V_H_n = np.sum(overlap * Vrho).real
        V_H_diag_est.append(V_H_n)
    
    print_fn(f"\n5. ESTIMATED HARTREE MATRIX ELEMENTS (first 5 bands at k=0):")
    print_fn(f"   Band    V_H (ISDF, Ry)")
    for n, vh in enumerate(V_H_diag_est):
        print_fn(f"   {n:>4}    {vh:>12.4f}")
    
    # 6. Suggest correction factor
    # The error is approximately: V_H_code / V_H_correct ~ n_rtot / n_rmu / (some factor)
    print_fn(f"\n6. POSSIBLE CORRECTION FACTORS:")
    print_fn(f"   n_rtot / n_rmu = {n_rtot / n_rmu:.2f}")
    print_fn(f"   n_rtot / cell_volume = {n_rtot / cell_volume:.2f}")
    print_fn(f"   sqrt(n_rtot) = {np.sqrt(n_rtot):.2f}")
    print_fn(f"   1/sqrt(cell_volume) = {1/np.sqrt(cell_volume):.4f}")
    
    # Based on the observed ~20x error, the correction might involve:
    # dividing by n_rmu and/or multiplying by cell_volume
    print_fn(f"\n   If V_H is ~20x too large, check if:")
    print_fn(f"   - Missing factor of 1/n_rmu = {1/n_rmu:.6f}")  
    print_fn(f"   - Missing factor of cell_volume/n_rtot = {cell_volume/n_rtot:.6f}")
    print_fn(f"   - Wavefunctions need rescaling by 1/sqrt(n_rtot)")
    
    print_fn("\n" + "="*70)
    
    return {
        'avg_psi_sq': avg_psi_sq,
        'sum_rho': np.sum(rho_mu),
        'V0_trace': V0_trace,
        'V_H_diag_est': V_H_diag_est,
    }


def compare_hartree(
    psi_G_k,        # (nk, nb, nspinor, nx, ny, nz)
    psi_rmu_val,    # (nk, nb_occ, nspinor, n_rmu)
    psi_rmu_sig,    # (nk, nb_sig, nspinor, n_rmu)
    V0_munu,        # (n_rmu, n_rmu)
    bdot,           # (3, 3)
    cell_volume,    # float
    nk_tot,         # int
    Gk_crys_all,    # list of (nG, 3) arrays
    print_fn=print,
):
    """Compare Hartree from both methods and print diagnostics."""
    
    print_fn("\n" + "="*70)
    print_fn("HARTREE COMPARISON: Standard DFT vs ISDF")
    print_fn("="*70)
    
    # Standard DFT approach
    rho_r, V_H_r = compute_hartree_standard(
        psi_G_k[:, :psi_rmu_val.shape[1]],  # Use same bands as valence
        bdot, cell_volume, nk_tot
    )
    V_H_diag_std = compute_hartree_mtxel_standard(
        psi_G_k[:, :psi_rmu_sig.shape[1]],  # Use sigma window
        V_H_r, cell_volume, Gk_crys_all
    )
    
    # ISDF approach
    rho_mu, Vrho_mu, V_H_diag_isdf = compute_hartree_isdf(
        psi_rmu_val, psi_rmu_sig, V0_munu, nk_tot
    )
    
    print_fn(f"\n1. DENSITY COMPARISON:")
    print_fn(f"   Standard ρ(r): grid={rho_r.shape}, total={float(jnp.sum(rho_r)):.6f}")
    print_fn(f"   ISDF ρ_μ: n_rmu={rho_mu.shape[0]}, total={float(jnp.sum(rho_mu)):.6f}")
    
    print_fn(f"\n2. HARTREE POTENTIAL:")
    print_fn(f"   Standard V_H(r): min={float(V_H_r.min()):.4f}, max={float(V_H_r.max()):.4f}")
    print_fn(f"   ISDF [V@ρ]_μ: min={float(Vrho_mu.real.min()):.4f}, max={float(Vrho_mu.real.max()):.4f}")
    
    print_fn(f"\n3. DIAGONAL MATRIX ELEMENTS ⟨n|V_H|n⟩ (Ry):")
    print_fn(f"   {'Band':>6} {'Standard':>12} {'ISDF':>12} {'Ratio':>10} {'Diff':>10}")
    print_fn("   " + "-"*54)
    
    n_compare = min(10, len(V_H_diag_std), len(V_H_diag_isdf))
    for n in range(n_compare):
        std_val = float(V_H_diag_std[n])
        isdf_val = float(V_H_diag_isdf[n])
        ratio = isdf_val / std_val if abs(std_val) > 1e-10 else float('nan')
        diff = isdf_val - std_val
        print_fn(f"   {n:>6} {std_val:>12.4f} {isdf_val:>12.4f} {ratio:>10.4f} {diff:>10.4f}")
    
    print_fn("\n4. V0_MUNU STATISTICS:")
    print_fn(f"   Shape: {V0_munu.shape}")
    print_fn(f"   Trace: {float(jnp.trace(V0_munu).real):.4f}")
    print_fn(f"   Max |element|: {float(jnp.abs(V0_munu).max()):.4f}")
    print_fn(f"   Frobenius norm: {float(jnp.linalg.norm(V0_munu)):.4f}")
    
    # Check if ratio is approximately constant
    ratios = np.array([float(V_H_diag_isdf[n] / V_H_diag_std[n]) 
                       for n in range(n_compare) if abs(float(V_H_diag_std[n])) > 1e-10])
    if len(ratios) > 1:
        print_fn(f"\n5. RATIO ANALYSIS:")
        print_fn(f"   Mean ratio: {ratios.mean():.4f}")
        print_fn(f"   Std ratio: {ratios.std():.4f}")
        print_fn(f"   If std is small, error is pure multiplicative scaling")
        print_fn(f"   If std is large, error has band-dependent component")
    
    print_fn("="*70 + "\n")
    
    return {
        'rho_r': rho_r,
        'V_H_r': V_H_r,
        'rho_mu': rho_mu,
        'Vrho_mu': Vrho_mu,
        'V_H_diag_std': V_H_diag_std,
        'V_H_diag_isdf': V_H_diag_isdf,
    }


"""Diagnostic to check if head correction is contaminating V0_noG0_munu for Hartree.

Run this from within cohsex_jax to compare v_q0_noG0_munu with/without head.
"""

import numpy as np
import jax.numpy as jnp


def diagnose_hartree_head(
    V_qmunu_before_head,    # V_qmunu[0,0,0,0,0,0] BEFORE head injection
    v_q0_noG0_munu,         # The array passed to compute_sigma_pipeline_jax
    G0_mu_nu,               # ζ_μ(G=0) vector used for head correction
    vc0_mean,               # The averaged v(q=0,G=0) 
    cell_volume,            # Cell volume in atomic units
    psi_val_rmu,            # (nk, nb_val, ns, n_rmu) valence wavefunctions at centroids
    psi_sig_rmu,            # (nk, nb_sig, ns, n_rmu) sigma wavefunctions at centroids  
    nk_tot,                 # Number of k-points
    print_fn=print,
):
    """Check if the Hartree V0 has spurious head contributions."""
    
    print_fn("\n" + "="*70)
    print_fn("HARTREE HEAD CONTAMINATION DIAGNOSTIC")
    print_fn("="*70)
    
    # 1. Check if v_q0_noG0_munu == V_qmunu_before_head
    V_before = np.asarray(V_qmunu_before_head)
    V0 = np.asarray(v_q0_noG0_munu)
    diff = np.linalg.norm(V0 - V_before)
    print_fn(f"\n1. V0_noG0 vs V_qmunu[q=0] before head injection:")
    print_fn(f"   ||V0_noG0 - V_qmunu_before|| = {diff:.3e}")
    if diff > 1e-10:
        print_fn(f"   ⚠️  MISMATCH - these should be identical!")
    
    # 2. Compute what the head correction looks like
    g0 = np.asarray(G0_mu_nu)
    vol_scale = 1.0 / float(cell_volume)
    head_outer = np.outer(g0.conj(), g0)
    head_correction = float(vc0_mean.real) * vol_scale * head_outer
    
    print_fn(f"\n2. Head correction analysis:")
    print_fn(f"   vc0_mean = {float(vc0_mean.real):.4f} (atomic units)")
    print_fn(f"   vol_scale = 1/Ω = {vol_scale:.6e}")
    print_fn(f"   ||ζ(G=0)||² = {np.sum(np.abs(g0)**2):.4f}")
    print_fn(f"   ||head_correction|| = {np.linalg.norm(head_correction):.4e}")
    print_fn(f"   Tr(head_correction) = {np.trace(head_correction).real:.4e}")
    
    # 3. Check if V0 contains any head-like contribution
    # Project V0 onto the head direction
    v0_proj_head = np.abs(np.vdot(V0, head_outer)) / (np.linalg.norm(V0) * np.linalg.norm(head_outer))
    print_fn(f"\n3. V0 projection onto head direction:")
    print_fn(f"   cos(angle) = {v0_proj_head:.6f}")
    print_fn(f"   (1.0 = parallel, 0.0 = orthogonal)")
    
    # 4. Compute Hartree with and without head contamination
    psi_val = np.asarray(psi_val_rmu)
    psi_sig = np.asarray(psi_sig_rmu)
    
    # Density
    rho_mu = np.sum(np.abs(psi_val)**2, axis=(0, 1, 2)) / nk_tot
    
    # Hartree with current V0
    Vrho_current = V0 @ rho_mu
    
    # Hypothetical Hartree if head was added to V0
    V0_with_head = V0 + head_correction
    Vrho_with_head = V0_with_head @ rho_mu
    
    print_fn(f"\n4. Effect of head on Hartree:")
    print_fn(f"   ||V0 @ ρ|| = {np.linalg.norm(Vrho_current):.4f}")
    print_fn(f"   ||(V0+head) @ ρ|| = {np.linalg.norm(Vrho_with_head):.4f}")
    print_fn(f"   ||head @ ρ|| = {np.linalg.norm(head_correction @ rho_mu):.4f}")
    
    # 5. Matrix element comparison for k=0
    psi_k0 = psi_sig[0]  # (nb, ns, n_rmu)
    nb_sig = psi_k0.shape[0]
    
    print_fn(f"\n5. Diagonal Hartree matrix elements at k=0:")
    print_fn(f"   {'Band':>6}  {'Current':>12}  {'If Head Added':>14}  {'Difference':>12}")
    print_fn(f"   {'-'*6}  {'-'*12}  {'-'*14}  {'-'*12}")
    
    for n in range(min(nb_sig, 12)):
        psi_n = psi_k0[n]  # (ns, n_rmu)
        overlap = np.sum(np.abs(psi_n)**2, axis=0)  # (n_rmu,)
        
        V_H_current = np.sum(overlap * Vrho_current).real
        V_H_with_head = np.sum(overlap * Vrho_with_head).real
        delta = V_H_with_head - V_H_current
        
        print_fn(f"   {n:>6}  {V_H_current:>12.4f}  {V_H_with_head:>14.4f}  {delta:>12.4f}")
    
    # 6. Check if the head-induced error matches the observed pattern
    print_fn(f"\n6. Head-induced error pattern:")
    
    errors = []
    for n in range(nb_sig):
        psi_n = psi_k0[n]
        overlap = np.sum(np.abs(psi_n)**2, axis=0)
        delta = np.sum(overlap * (head_correction @ rho_mu)).real
        errors.append(delta)
    
    errors = np.array(errors)
    print_fn(f"   Min head-induced error: {errors.min():.4f} Ry")
    print_fn(f"   Max head-induced error: {errors.max():.4f} Ry")
    print_fn(f"   Mean head-induced error: {errors.mean():.4f} Ry")
    
    if errors.max() > 0.1:
        print_fn(f"\n   ⚠️  Head correction would add ~{errors.mean():.1f} Ry additive error!")
        print_fn(f"   This matches the observed 'additive ~7 Ry' pattern!")
    
    print_fn("\n" + "="*70)
    
    return {
        'diff_before_head': diff,
        'head_correction_norm': np.linalg.norm(head_correction),
        'head_induced_errors': errors,
    }


def check_v_munu_g0_component(
    zeta_q_at_q0,  # (n_rmu, n_rtot) or (n_rmu, nx, ny, nz)
    print_fn=print,
):
    """Check if ζ(G=0) component contributes to v_munu despite Coulomb being zero there."""
    
    zeta = np.asarray(zeta_q_at_q0)
    if zeta.ndim == 2:
        # Assume already in G-space, flattened
        zeta_G0 = zeta[:, 0]  # First component is G=0
    else:
        # Real-space, need FFT
        zeta_G = np.fft.fftn(zeta, axes=(-3, -2, -1))
        zeta_G0 = zeta_G[:, 0, 0, 0]
    
    print_fn(f"\n7. ζ(G=0) analysis at q=0:")
    print_fn(f"   ||ζ(G=0)|| = {np.linalg.norm(zeta_G0):.4f}")
    print_fn(f"   Max |ζ_μ(G=0)| = {np.abs(zeta_G0).max():.4f}")
    print_fn(f"   This component SHOULD be zeroed in v_munu since v(G=0)=0")
    
    # But check if it contributes due to numerical issues
    outer = np.outer(zeta_G0.conj(), zeta_G0)
    print_fn(f"   ||ζ(G=0) ⊗ ζ*(G=0)|| = {np.linalg.norm(outer):.4e}")
    
    return zeta_G0


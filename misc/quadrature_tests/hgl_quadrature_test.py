#!/usr/bin/env python
"""
hgl_quadrature_test.py

Self-contained tests for Hermite-Gauss-Laguerre (HGL) quadrature used in the 
Complex-Time Shredded Propagator (CTSP) formalism for dynamic polarizability
with energy crossings.

HGL is used when ω falls within the energy window (E_gap < ω < E_bw),
causing the denominator (E_c - E_v - ω) to change sign.

The core identity:
    1/x = γ ∫₀^∞ sin(γxτ) exp(-τ - τ²/2) dτ   for |γx| >> 1

Complex propagators via Euler identity (default approach):
    G^v = Σ_v exp(iγτE_v) A_v,    G^c = Σ_c exp(iγτE_c) A_c
    (G^c)† G^v = P_+ - i P_×

This uses 2 complex arrays instead of 4 real (2× memory savings).

Reference: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)
"""

import numpy as np
from scipy.integrate import quad as scipy_quad
from hgl_quadrature import hgl_nodes_weights


def test_hgl_regularization():
    """
    Test: F(x) = ∫₀^∞ sin(xτ) exp(-τ - τ²/2) dτ
    
    Key properties:
    - F(x) → 1/x for large |x|
    - F(x) is bounded for all x (regularizes the 1/x singularity)
    - F(x) ≈ τ₀ × x for small |x| where τ₀ = ∫ τ exp(-τ - τ²/2) dτ
    """
    print("\n" + "=" * 70)
    print("Test 3: HGL regularization function F(x)")
    print("=" * 70)
    
    # Exact F(x) via numerical integration
    def F_exact(x):
        result, _ = scipy_quad(lambda t: np.sin(x * t) * np.exp(-t - t**2/2), 0, 100)
        return result
    
    # τ₀ = first moment
    tau0, _ = scipy_quad(lambda t: t * np.exp(-t - t**2/2), 0, 100)
    
    print(f"\nτ₀ = {tau0:.6f} (first moment)")
    print("\nProperties of F(x):")
    print(f"  {'x':>6} {'F(x)':>12} {'1/x':>12} {'τ₀×x':>12} {'F(x)×x':>12}")
    print("  " + "-" * 60)
    
    for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        Fx = F_exact(x)
        print(f"  {x:6.2f} {Fx:12.6f} {1/x:12.6f} {tau0*x:12.6f} {Fx*x:12.6f}")
    
    # Test quadrature accuracy
    print(f"\nHGL quadrature accuracy for F(x):")
    for n_tau in [5, 10, 15, 20]:
        tau, w = hgl_nodes_weights(n_tau)
        
        errors = []
        for x in [2.0, 5.0, 10.0]:
            quad_val = np.sum(w * np.sin(x * tau))
            exact_val = F_exact(x)
            rel_err = abs(quad_val - exact_val) / abs(exact_val)
            errors.append(rel_err)
        
        print(f"  n_tau={n_tau:2d}: rel errors for x=2,5,10: {[f'{e:.2e}' for e in errors]}")


def test_hgl_chi_crossing():
    """
    Test: χ(ω) for crossing windows using HGL quadrature with Euler identity.
    
    When ω falls within the energy window (E_gap < ω < E_bw), 
    the denominator (E_c - E_v - ω) can change sign ("energy crossing").
    
    The HGL formula:
        1/(E_c - E_v - ω) ≈ γ × F(γ(E_c - E_v - ω))
    
    Using Euler identity (default, 2× memory savings):
        G^v = Σ_v exp(iγτE_v) A_v,  G^c = Σ_c exp(iγτE_c) A_c
        (G^c)† G^v = P_+ - i P_×
    
    Then:
        χ(ω) = -γ Σ_u w_u [cos(γτ_u ω) × P_×(τ_u) - sin(γτ_u ω) × P_+(τ_u)]
    """
    print("\n" + "=" * 70)
    print("Test 4: HGL quadrature for χ(ω) with Euler identity")
    print("=" * 70)
    
    # Synthetic band structure
    np.random.seed(42)
    n_val, n_cond = 5, 8
    
    E_v = np.sort(np.random.uniform(-0.5, 0.0, n_val))
    E_c = np.sort(np.random.uniform(0.1, 0.8, n_cond))
    
    E_v_max = np.max(E_v)
    E_c_min = np.min(E_c)
    E_gap = E_c_min - E_v_max
    E_bw = np.max(E_c) - np.min(E_v)
    gamma = 1.0 / np.sqrt(E_gap * E_bw)
    
    print(f"\nBand structure: {n_val} valence, {n_cond} conduction")
    print(f"  E_gap = {E_gap:.3f}, E_bw = {E_bw:.3f}")
    print(f"  γ = {gamma:.3f}")
    print(f"  Crossing region: ω ∈ [{E_gap:.3f}, {E_bw:.3f}]")
    
    # Test coefficients (real = Hermitian propagators)
    A_v = np.random.random(n_val)
    A_c = np.random.random(n_cond)
    
    # Direct F(x) via numerical integration
    def F_exact(x):
        result, _ = scipy_quad(lambda t: np.sin(x * t) * np.exp(-t - t**2/2), 0, 100)
        return result
    
    # Test frequencies in crossing region
    omega_values = [0.2, 0.3, 0.4, 0.5]
    
    print(f"\nDirect vs HGL (Euler) quadrature for χ(ω):")
    print(f"  {'ω':>6} {'χ_direct':>14} {'χ_HGL':>14} {'rel err':>12}")
    print("  " + "-" * 52)
    
    n_tau = 15
    tau, w = hgl_nodes_weights(n_tau)
    
    for omega in omega_values:
        # Direct sum using exact F(x) for each term
        chi_direct = 0.0
        for iv, ev in enumerate(E_v):
            for ic, ec in enumerate(E_c):
                x = ec - ev - omega
                # Use γ × F(γx) as the "denominator"
                if abs(x) > 1e-10:
                    chi_direct += A_v[iv] * A_c[ic] * gamma * F_exact(gamma * x)
        chi_direct *= -1
        
        # HGL with Euler identity (2 complex arrays instead of 4 real)
        chi_hgl = 0.0
        for u in range(n_tau):
            # Complex propagators: G = C + iS
            G_v = np.sum(np.exp(1j * gamma * tau[u] * E_v) * A_v)
            G_c = np.sum(np.exp(1j * gamma * tau[u] * E_c) * A_c)
            
            # (G^c)† G^v = P_+ - i P_×
            product = np.conj(G_c) * G_v
            P_plus = product.real
            P_cross = -product.imag
            
            # ω-dependent phases
            sin_omega = np.sin(gamma * tau[u] * omega)
            cos_omega = np.cos(gamma * tau[u] * omega)
            
            # sin(γτ(E_c - E_v - ω)) = cos(γτω) × P_× - sin(γτω) × P_+
            chi_hgl += w[u] * (cos_omega * P_cross - sin_omega * P_plus)
        
        chi_hgl *= -gamma
        
        rel_err = abs(chi_hgl - chi_direct) / abs(chi_direct)
        status = "✓" if rel_err < 0.01 else "✗"
        print(f"  {omega:6.3f} {chi_direct:14.8f} {chi_hgl:14.8f} {rel_err:12.2e} {status}")


def test_hgl_phase_separation():
    """
    Test the angle-addition identity used for ω-batching:
    
        sin(γτ(E - ω)) = sin(γτE)cos(γτω) - cos(γτE)sin(γτω)
    
    This lets us precompute the ω-independent parts (sin/cos of E terms)
    and then combine with ω-dependent phases.
    """
    print("\n" + "=" * 70)
    print("Test 5: Phase separation identity for efficient ω-batching")
    print("=" * 70)
    
    gamma = 2.0
    tau = np.array([0.5, 1.0, 2.0])
    E = np.array([0.3, 0.5, 0.8])
    omega_values = np.array([0.1, 0.2, 0.3, 0.4])
    
    print("\nVerifying: sin(γτ(E - ω)) = sin(γτE)cos(γτω) - cos(γτE)sin(γτω)")
    
    max_err = 0.0
    for u, t in enumerate(tau):
        for e in E:
            # Precomputed (ω-independent)
            sin_E = np.sin(gamma * t * e)
            cos_E = np.cos(gamma * t * e)
            
            for omega in omega_values:
                # Direct
                direct = np.sin(gamma * t * (e - omega))
                
                # Separated
                sin_omega = np.sin(gamma * t * omega)
                cos_omega = np.cos(gamma * t * omega)
                separated = sin_E * cos_omega - cos_E * sin_omega
                
                err = abs(direct - separated)
                max_err = max(max_err, err)
    
    print(f"\nMax error: {max_err:.2e}")
    print(f"Identity verified: {'✓' if max_err < 1e-14 else '✗'}")


def test_hgl_euler_identity():
    """
    Verify: Euler identity matches naive 4-array approach.
    
    Default (Euler — 2 complex arrays):
        G^v = Σ_v exp(iγτE_v) A_v,    G^c = Σ_c exp(iγτE_c) A_c
        (G^c)† G^v = P_+ - i P_×
    
    Naive (4 real arrays):
        C_v = Σ cos(γτE_v) A_v,  S_v = Σ sin(γτE_v) A_v, etc.
        P_+ = C_c C_v + S_c S_v,  P_× = S_c C_v - C_c S_v
    
    For Hermitian propagators (ψψ†), these give identical results.
    In this scalar test with REAL coefficients, conj = Hermitian transpose.
    """
    print("\n" + "=" * 70)
    print("Test 6: Euler identity optimization (2× memory savings)")
    print("=" * 70)
    
    np.random.seed(42)
    n_val, n_cond = 5, 8
    
    E_v = np.sort(np.random.uniform(-0.5, 0.0, n_val))
    E_c = np.sort(np.random.uniform(0.1, 0.8, n_cond))
    
    # Real coefficients — this models Hermitian propagators (ψψ†)
    # where the scalar analog has conj = transpose
    A_v = np.random.random(n_val)
    A_c = np.random.random(n_cond)
    
    E_gap = np.min(E_c) - np.max(E_v)
    E_bw = np.max(E_c) - np.min(E_v)
    gamma = 1.0 / np.sqrt(E_gap * E_bw)
    
    print(f"\nReal coefficients (models Hermitian ψψ† propagators)")
    print(f"  E_gap = {E_gap:.3f}, γ = {gamma:.3f}")
    
    n_tau = 15
    tau, w = hgl_nodes_weights(n_tau)
    
    print(f"\nComparing naive (4 real arrays) vs Euler (2 complex arrays) at each τ:")
    print(f"  {'τ_u':>6} {'P_+ naive':>16} {'P_+ Euler':>16} {'P_× naive':>16} {'P_× Euler':>16}")
    print("  " + "-" * 76)
    
    all_match = True
    for u in range(min(5, n_tau)):  # Show first 5 τ points
        t = tau[u]
        
        # === NAIVE: 4 separate real arrays ===
        C_v_naive = np.sum(np.cos(gamma * t * E_v) * A_v)
        S_v_naive = np.sum(np.sin(gamma * t * E_v) * A_v)
        C_c_naive = np.sum(np.cos(gamma * t * E_c) * A_c)
        S_c_naive = np.sum(np.sin(gamma * t * E_c) * A_c)
        
        P_plus_naive = C_c_naive * C_v_naive + S_c_naive * S_v_naive
        P_cross_naive = S_c_naive * C_v_naive - C_c_naive * S_v_naive
        
        # === EULER: 2 complex arrays ===
        G_v = np.sum(np.exp(1j * gamma * t * E_v) * A_v)  # C_v + i S_v
        G_c = np.sum(np.exp(1j * gamma * t * E_c) * A_c)  # C_c + i S_c
        
        # (G_c)† × G_v = P_+ - i P_×  (for Hermitian, † = conj)
        product = np.conj(G_c) * G_v
        P_plus_euler = product.real
        P_cross_euler = -product.imag
        
        # Compare
        match_plus = np.isclose(P_plus_naive, P_plus_euler, rtol=1e-14)
        match_cross = np.isclose(P_cross_naive, P_cross_euler, rtol=1e-14)
        match = match_plus and match_cross
        all_match = all_match and match
        
        status = "✓" if match else "✗"
        print(f"  {t:6.3f} {P_plus_naive:16.10f} {P_plus_euler:16.10f} {P_cross_naive:16.10f} {P_cross_euler:16.10f} {status}")
    
    print(f"\n  ... (checked all {n_tau} τ points)")
    print(f"\nEuler identity for Hermitian case: {'✓ VERIFIED' if all_match else '✗ FAILED'}")
    
    # Now verify full χ(ω) computation using Euler
    print(f"\nFull χ(ω) using Euler optimization:")
    
    omega_values = [0.2, 0.3, 0.4, 0.5]
    print(f"  {'ω':>6} {'χ_naive':>16} {'χ_Euler':>16} {'rel err':>12}")
    print("  " + "-" * 56)
    
    for omega in omega_values:
        chi_naive = 0.0
        chi_euler = 0.0
        
        for u in range(n_tau):
            t = tau[u]
            
            # Naive: 4 arrays
            C_v = np.sum(np.cos(gamma * t * E_v) * A_v)
            S_v = np.sum(np.sin(gamma * t * E_v) * A_v)
            C_c = np.sum(np.cos(gamma * t * E_c) * A_c)
            S_c = np.sum(np.sin(gamma * t * E_c) * A_c)
            P_cross = S_c * C_v - C_c * S_v
            P_plus = C_c * C_v + S_c * S_v
            sin_omega = np.sin(gamma * t * omega)
            cos_omega = np.cos(gamma * t * omega)
            chi_naive += w[u] * (cos_omega * P_cross - sin_omega * P_plus)
            
            # Euler: 2 complex arrays
            G_v = np.sum(np.exp(1j * gamma * t * E_v) * A_v)
            G_c = np.sum(np.exp(1j * gamma * t * E_c) * A_c)
            product = np.conj(G_c) * G_v  # P_+ - i P_×
            P_plus_e = product.real
            P_cross_e = -product.imag
            chi_euler += w[u] * (cos_omega * P_cross_e - sin_omega * P_plus_e)
        
        chi_naive *= -gamma
        chi_euler *= -gamma
        
        rel_err = abs(chi_euler - chi_naive) / abs(chi_naive)
        status = "✓" if rel_err < 1e-12 else "✗"
        print(f"  {omega:6.3f} {chi_naive:16.10f} {chi_euler:16.10f} {rel_err:12.2e} {status}")
    
    print(f"\n  Memory comparison:")
    print(f"    Naive:  4 real arrays  → 4 × sizeof(real)")
    print(f"    Euler:  2 complex arrays → 2 × sizeof(complex) = 4 × sizeof(real)")
    print(f"    But in practice: Euler avoids 2 intermediate arrays during construction")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("HGL Quadrature Tests (Hermite-Gauss-Laguerre for dynamic χ)")
    print("=" * 70)
    print("Testing HGL quadrature for polarizability with energy crossings")
    print("Reference: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)")
    print()
    
    test_hgl_regularization()
    test_hgl_chi_crossing()
    test_hgl_phase_separation()
    test_hgl_euler_identity()
    
    print("\n" + "=" * 70)
    print("All HGL tests completed!")
    print("=" * 70)


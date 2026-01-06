#!/usr/bin/env python
"""
gl_quadrature_test.py

Self-contained tests for Gauss-Laguerre (GL) quadrature used in the 
Complex-Time Shredded Propagator (CTSP) formalism for static polarizability.

The core identity:
    1/(E_c - E_v) = ζ ∫₀^∞ exp(-ζ(E_c - E_v)τ) dτ
    
With GL weight exp(-τ):
    1/s = ζ ∫₀^∞ exp(-(ζs - 1)τ) exp(-τ) dτ
        ≈ ζ Σ_u w_u exp(-(ζs - 1)τ_u)

For static polarizability (separable form):
    χ = -Σ_{vc} A_v A_c / (E_c - E_v)
      = -ζ Σ_u w_u exp(-(ζE_gap - 1)τ_u) × ρ_c(τ_u) × ρ̄_v(τ_u)

where:
    ρ_c(τ) = Σ_c exp(-ζτ(E_c - E_c_min)) A_c
    ρ̄_v(τ) = Σ_v exp(-ζτ(E_v_max - E_v)) A_v
    ζ = 1/√(E_gap × E_bw)

Reference: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)
"""

import numpy as np
from scipy.special import roots_laguerre


def test_gl_single_denominator():
    """
    Test: 1/s ≈ ζ Σ_u w_u exp(-(ζs - 1)τ_u)
    
    Verifies the fundamental Laplace identity for a single denominator.
    """
    print("=" * 70)
    print("Test 1: GL quadrature for single denominator 1/s")
    print("=" * 70)
    
    s_values = [0.5, 1.0, 2.0, 5.0]
    
    for n_tau in [5, 10, 15]:
        tau, w = roots_laguerre(n_tau)
        
        print(f"\nn_tau = {n_tau}:")
        print(f"  {'s':>6} {'exact':>12} {'GL quad':>12} {'rel err':>12}")
        print("  " + "-" * 48)
        
        for s in s_values:
            zeta = 1.0
            exact = 1.0 / s
            quad_val = zeta * np.sum(w * np.exp(-(zeta * s - 1) * tau))
            rel_err = abs(quad_val - exact) / abs(exact)
            print(f"  {s:6.2f} {exact:12.6f} {quad_val:12.6f} {rel_err:12.2e}")


def test_gl_optimal_scaling():
    """
    Test the optimal scaling ζ = 1/√(s_min × s_max) for a range of denominators.
    
    Without optimal scaling, GL struggles with wide energy ranges.
    With optimal scaling, accuracy is uniform across the range.
    """
    print("\n" + "=" * 70)
    print("Test 2: GL quadrature with optimal ζ scaling")
    print("=" * 70)
    
    # Energy range
    s_min, s_max = 0.1, 10.0
    s_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    print(f"\nEnergy range: s ∈ [{s_min}, {s_max}]")
    
    # Without optimal scaling
    print(f"\nWithout scaling (ζ = 1):")
    zeta = 1.0
    for n_tau in [10, 20]:
        tau, w = roots_laguerre(n_tau)
        errors = []
        for s in s_values:
            exact = 1.0 / s
            quad = zeta * np.sum(w * np.exp(-(zeta * s - 1) * tau))
            errors.append(abs(quad - exact) / abs(exact))
        print(f"  n_tau={n_tau:2d}: max rel err = {max(errors):.2e}")
    
    # With optimal scaling
    print(f"\nWith optimal scaling (ζ = 1/√(s_min × s_max) = {1/np.sqrt(s_min * s_max):.3f}):")
    zeta = 1.0 / np.sqrt(s_min * s_max)
    for n_tau in [10, 20]:
        tau, w = roots_laguerre(n_tau)
        errors = []
        for s in s_values:
            exact = 1.0 / s
            quad = zeta * np.sum(w * np.exp(-(zeta * s - 1) * tau))
            errors.append(abs(quad - exact) / abs(exact))
        print(f"  n_tau={n_tau:2d}: max rel err = {max(errors):.2e}")


def test_gl_chi_separable():
    """
    Test: χ = -Σ_{vc} A_v A_c / (E_c - E_v) via separable GL quadrature
    
    The key insight is that the double sum factorizes:
        χ = -ζ Σ_u w_u × gap_factor(τ_u) × ρ_c(τ_u) × ρ̄_v(τ_u)
    
    This is O(N_τ × (N_v + N_c)) instead of O(N_v × N_c).
    """
    print("\n" + "=" * 70)
    print("Test 3: GL quadrature for χ = -Σ A_v A_c / (E_c - E_v)")
    print("=" * 70)
    
    np.random.seed(42)
    n_val, n_cond = 10, 15
    
    E_v = np.sort(np.random.uniform(-1.0, 0.0, n_val))
    E_c = np.sort(np.random.uniform(0.2, 2.0, n_cond))
    
    E_v_max = np.max(E_v)
    E_c_min = np.min(E_c)
    E_gap = E_c_min - E_v_max
    E_bw = np.max(E_c) - np.min(E_v)
    zeta = 1.0 / np.sqrt(E_gap * E_bw)
    
    print(f"\nBand structure: {n_val} valence, {n_cond} conduction")
    print(f"  E_v range: [{np.min(E_v):.3f}, {E_v_max:.3f}]")
    print(f"  E_c range: [{E_c_min:.3f}, {np.max(E_c):.3f}]")
    print(f"  E_gap = {E_gap:.3f}, E_bw = {E_bw:.3f}, ζ = {zeta:.3f}")
    
    coeff_sets = [
        ("uniform", np.ones(n_val), np.ones(n_cond)),
        ("linear", np.linspace(1, 2, n_val), np.linspace(1, 2, n_cond)),
        ("random", np.random.random(n_val), np.random.random(n_cond)),
    ]
    
    for name, A_v, A_c in coeff_sets:
        print(f"\nCoefficients: {name}")
        
        # Direct O(N_v × N_c) sum
        denom = E_c[np.newaxis, :] - E_v[:, np.newaxis]
        chi_direct = -np.sum(A_v[:, None] * A_c[None, :] / denom)
        
        # Separable GL quadrature
        for n_tau in [5, 10, 15, 20]:
            tau, w = roots_laguerre(n_tau)
            
            # Propagators: O(N_τ × N_c) and O(N_τ × N_v)
            rho_c = np.exp(-zeta * tau[:, None] * (E_c - E_c_min)) @ A_c
            rho_v = np.exp(-zeta * tau[:, None] * (E_v_max - E_v)) @ A_v
            
            # Gap factor
            gap_factor = np.exp(-(zeta * E_gap - 1) * tau)
            
            # Quadrature: O(N_τ)
            chi_gl = -zeta * np.sum(w * gap_factor * rho_c * rho_v)
            
            rel_err = abs(chi_gl - chi_direct) / abs(chi_direct)
            status = "✓" if rel_err < 0.05 else "✗"
            print(f"  n_tau={n_tau:2d}: χ_direct={chi_direct:12.6f}, χ_GL={chi_gl:12.6f}, err={rel_err:.2e} {status}")


def test_gl_quadrature_points():
    """
    Test the empirical formula for number of quadrature points:
        N_τ = √(E_bw/E_gap) × (0.4 - 0.3 ln ε)
    
    where ε is the target relative error.
    """
    print("\n" + "=" * 70)
    print("Test 4: Empirical formula for N_τ")
    print("=" * 70)
    
    np.random.seed(123)
    
    # Test several gap/bandwidth ratios
    test_cases = [
        (0.1, 1.0, "narrow gap"),
        (0.5, 2.0, "moderate gap"),
        (1.0, 5.0, "wide bandwidth"),
    ]
    
    for E_gap, E_bw, label in test_cases:
        print(f"\n{label}: E_gap={E_gap}, E_bw={E_bw}")
        
        # Generate synthetic bands
        n_val, n_cond = 8, 12
        E_v = np.sort(np.random.uniform(-E_bw/2, -E_gap/2, n_val))
        E_c = np.sort(np.random.uniform(E_gap/2, E_bw/2, n_cond))
        
        # Shift to get exact gap
        E_v = E_v - np.max(E_v)
        E_c = E_c - np.min(E_c) + E_gap
        
        actual_gap = np.min(E_c) - np.max(E_v)
        actual_bw = np.max(E_c) - np.min(E_v)
        zeta = 1.0 / np.sqrt(actual_gap * actual_bw)
        
        A_v = np.random.random(n_val)
        A_c = np.random.random(n_cond)
        
        # Direct sum
        denom = E_c[np.newaxis, :] - E_v[:, np.newaxis]
        chi_direct = -np.sum(A_v[:, None] * A_c[None, :] / denom)
        
        # Test empirical formula at different error targets
        for eps_target in [0.01, 0.001]:
            n_tau_formula = np.sqrt(actual_bw / actual_gap) * (0.4 - 0.3 * np.log(eps_target))
            n_tau = max(3, int(np.ceil(n_tau_formula)))
            
            tau, w = roots_laguerre(n_tau)
            rho_c = np.exp(-zeta * tau[:, None] * (E_c - np.min(E_c))) @ A_c
            rho_v = np.exp(-zeta * tau[:, None] * (np.max(E_v) - E_v)) @ A_v
            gap_factor = np.exp(-(zeta * actual_gap - 1) * tau)
            chi_gl = -zeta * np.sum(w * gap_factor * rho_c * rho_v)
            
            actual_err = abs(chi_gl - chi_direct) / abs(chi_direct)
            status = "✓" if actual_err < eps_target else "✗"
            print(f"  ε={eps_target:.0e}: formula→n_τ={n_tau:2d}, actual_err={actual_err:.2e} {status}")


if __name__ == "__main__":
    print("GL Quadrature Tests (Gauss-Laguerre for static χ)")
    print("=" * 70)
    print("Reference: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)")
    print()
    
    test_gl_single_denominator()
    test_gl_optimal_scaling()
    test_gl_chi_separable()
    test_gl_quadrature_points()
    
    print("\n" + "=" * 70)
    print("All GL tests completed!")
    print("=" * 70)

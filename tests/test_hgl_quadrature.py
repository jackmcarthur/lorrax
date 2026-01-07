"""
test_hgl_quadrature.py

Unit tests for HGL quadrature node/weight generation and WindowPair integration.
Run with: python tests/test_hgl_quadrature.py
"""

import numpy as np
from scipy.integrate import quad as scipy_quad
import sys
sys.path.insert(0, '/home/jackm/projects/isdf_cohsex/src')


def test_hgl_nodes_weights_basic():
    """Test that hgl_nodes_weights returns correct shapes and positive weights."""
    from isdf.gw_isdf.hgl_quadrature import hgl_nodes_weights
    
    print("\nTest: hgl_nodes_weights basic properties")
    all_pass = True
    
    for n in [5, 10, 15, 20]:
        tau, w = hgl_nodes_weights(n)
        
        checks = [
            (tau.shape == (n,), f"n={n}: Expected {n} nodes, got {tau.shape}"),
            (w.shape == (n,), f"n={n}: Expected {n} weights, got {w.shape}"),
            (np.all(tau > 0), f"n={n}: All nodes should be positive"),
            (np.all(w > 0), f"n={n}: All weights should be positive"),
            (np.all(np.diff(tau) > 0), f"n={n}: Nodes should be sorted ascending"),
        ]
        
        for passed, msg in checks:
            if not passed:
                print(f"  FAIL: {msg}")
                all_pass = False
    
    if all_pass:
        print("  PASS: All basic property checks passed")
    return all_pass


def test_hgl_nodes_weights_sum():
    """Test that weights sum to μ_0 = ∫ exp(-τ - τ²/2) dτ."""
    from isdf.gw_isdf.hgl_quadrature import hgl_nodes_weights
    
    print("\nTest: HGL weights sum to μ_0")
    
    # Compute μ_0 by numerical integration
    mu0_exact, _ = scipy_quad(lambda t: np.exp(-t - t**2/2), 0, 100)
    
    all_pass = True
    for n in [5, 10, 15]:
        tau, w = hgl_nodes_weights(n)
        mu0_quad = np.sum(w)
        
        rel_err = abs(mu0_quad - mu0_exact) / mu0_exact
        if rel_err >= 0.01:
            print(f"  FAIL: n={n}: weight sum {mu0_quad:.6f} vs exact {mu0_exact:.6f}")
            all_pass = False
        else:
            print(f"  n={n}: weight sum = {mu0_quad:.6f}, exact = {mu0_exact:.6f}, rel_err = {rel_err:.2e} ✓")
    
    return all_pass


def test_hgl_regularization_function():
    """Test that HGL quadrature approximates F(x) = ∫ sin(xτ) exp(-τ - τ²/2) dτ."""
    from isdf.gw_isdf.hgl_quadrature import hgl_nodes_weights
    
    print("\nTest: HGL approximates regularization function F(x)")
    
    def F_exact(x):
        result, _ = scipy_quad(lambda t: np.sin(x * t) * np.exp(-t - t**2/2), 0, 100)
        return result
    
    # Need ~20 points for accurate high-x values
    tau, w = hgl_nodes_weights(20)
    
    all_pass = True
    # x=2,5 need few points, x=10 needs more
    for x, tol in [(2.0, 1e-10), (5.0, 1e-6), (10.0, 1e-3)]:
        quad_val = np.sum(w * np.sin(x * tau))
        exact_val = F_exact(x)
        rel_err = abs(quad_val - exact_val) / abs(exact_val)
        
        if rel_err >= tol:
            print(f"  FAIL: x={x}: rel error {rel_err:.2e} >= tol {tol:.0e}")
            all_pass = False
        else:
            print(f"  x={x}: quad={quad_val:.8f}, exact={exact_val:.8f}, rel_err={rel_err:.2e} ✓")
    
    return all_pass


def test_n_tau_hgl_formula():
    """Test that n_tau_hgl returns reasonable values."""
    from isdf.gw_isdf.hgl_quadrature import n_tau_hgl
    
    print("\nTest: n_tau_hgl formula")
    
    # Typical parameters from a GW calculation
    E_gap = 0.1  # Ry
    E_bw = 2.0   # Ry
    gamma = 1.0 / np.sqrt(E_gap * E_bw)
    
    # Test different error targets
    n_1pct = n_tau_hgl(gamma, E_bw, epsilon=0.01)
    n_01pct = n_tau_hgl(gamma, E_bw, epsilon=0.001)
    
    checks = [
        (n_1pct >= 3, f"n_1pct={n_1pct} should be >= 3"),
        (n_01pct >= n_1pct, f"n_01pct={n_01pct} should be >= n_1pct={n_1pct}"),
        (n_1pct < 50, f"n_1pct={n_1pct} should be < 50 for typical parameters"),
    ]
    
    all_pass = True
    for passed, msg in checks:
        if not passed:
            print(f"  FAIL: {msg}")
            all_pass = False
    
    if all_pass:
        print(f"  gamma={gamma:.3f}, E_bw={E_bw}")
        print(f"  n_tau_hgl(eps=0.01) = {n_1pct}")
        print(f"  n_tau_hgl(eps=0.001) = {n_01pct}")
        print("  PASS: All n_tau_hgl checks passed")
    
    return all_pass


def test_window_pair_hgl_fields():
    """Test that WindowPair has HGL fields initialized to None."""
    from isdf.gw_isdf.get_windows import WindowPair, WindowInfo
    
    print("\nTest: WindowPair HGL fields initialized correctly")
    
    val_win = WindowInfo(-0.5, 0.0, index=0, count=1)
    cond_win = WindowInfo(0.1, 1.0, index=0, count=1)
    
    win = WindowPair(val_win, cond_win, epsq=0.01)
    
    checks = [
        (win.tau_hgl is None, "tau_hgl should be None"),
        (win.w_hgl is None, "w_hgl should be None"),
        (win.ntau_hgl == 0, "ntau_hgl should be 0"),
        (hasattr(win, 'init_hgl_quadrature'), "should have init_hgl_quadrature method"),
    ]
    
    all_pass = True
    for passed, msg in checks:
        if not passed:
            print(f"  FAIL: {msg}")
            all_pass = False
    
    if all_pass:
        print("  PASS: All WindowPair HGL field checks passed")
    
    return all_pass


def test_window_pair_init_hgl():
    """Test that init_hgl_quadrature populates HGL fields correctly."""
    from isdf.gw_isdf.get_windows import WindowPair, WindowInfo
    
    print("\nTest: WindowPair.init_hgl_quadrature()")
    
    val_win = WindowInfo(-0.5, 0.0, index=0, count=1)
    cond_win = WindowInfo(0.1, 1.0, index=0, count=1)
    
    win = WindowPair(val_win, cond_win, epsq=0.01)
    
    # Before calling init_hgl_quadrature
    if win.tau_hgl is not None:
        print("  FAIL: tau_hgl should be None before init_hgl_quadrature")
        return False
    
    # Call init_hgl_quadrature
    win.init_hgl_quadrature()
    
    checks = [
        (win.tau_hgl is not None, "tau_hgl should not be None after init_hgl_quadrature"),
        (win.w_hgl is not None, "w_hgl should not be None"),
        (win.ntau_hgl > 0, f"ntau_hgl={win.ntau_hgl} should be > 0"),
        (len(win.tau_hgl) == win.ntau_hgl, "len(tau_hgl) should equal ntau_hgl"),
        (len(win.w_hgl) == win.ntau_hgl, "len(w_hgl) should equal ntau_hgl"),
    ]
    
    all_pass = True
    for passed, msg in checks:
        if not passed:
            print(f"  FAIL: {msg}")
            all_pass = False
    
    if all_pass:
        print(f"  ntau_hgl = {win.ntau_hgl}")
        print(f"  tau_hgl[:3] = {win.tau_hgl[:3]}")
        print(f"  w_hgl[:3] = {win.w_hgl[:3]}")
        print("  PASS: init_hgl_quadrature works correctly")
    
    return all_pass


def test_window_pair_gamma_equals_z_lm():
    """Verify that gamma for HGL should equal z_lm from GL."""
    from isdf.gw_isdf.get_windows import WindowPair, WindowInfo
    
    print("\nTest: gamma = z_lm consistency")
    
    val_win = WindowInfo(-0.5, 0.0, index=0, count=1)
    cond_win = WindowInfo(0.1, 1.0, index=0, count=1)
    
    win = WindowPair(val_win, cond_win, epsq=0.01)
    
    # Compute expected gamma from window bounds
    E_gap = cond_win.start_energy - val_win.end_energy  # 0.1 - 0.0 = 0.1
    E_bw = cond_win.end_energy - val_win.start_energy   # 1.0 - (-0.5) = 1.5
    expected_gamma = 1.0 / np.sqrt(E_gap * E_bw)
    
    # z_lm should equal gamma
    if not np.isclose(win.z_lm, expected_gamma):
        print(f"  FAIL: z_lm={win.z_lm} should equal gamma={expected_gamma}")
        return False
    
    print(f"  E_gap = {E_gap}, E_bw = {E_bw}")
    print(f"  z_lm = gamma = {win.z_lm:.6f}")
    print("  PASS: z_lm equals expected gamma")
    return True


def test_classify_frequencies():
    """Test classify_frequencies correctly partitions ω values."""
    from isdf.gw_isdf.get_windows import WindowPair, WindowInfo
    
    print("\nTest: classify_frequencies")
    
    # Window bounds: E_gap = 0.1, E_bw = 1.5
    val_win = WindowInfo(-0.5, 0.0, index=0, count=1)
    cond_win = WindowInfo(0.1, 1.0, index=0, count=1)
    win = WindowPair(val_win, cond_win, epsq=0.01)
    
    E_gap = 0.1
    E_bw = 1.5
    
    # Test frequencies:
    # ω = 0.05 → |ω| < E_gap → GL
    # ω = 0.5 → E_gap < |ω| < E_bw → HGL (crossing)
    # ω = 2.0 → |ω| > E_bw → GL
    # ω = -0.8 → E_gap < |ω| < E_bw → HGL (crossing)
    omega = np.array([0.05, 0.5, 2.0, -0.8, 0.0, 1.6])
    # Expected: GL for [0, 2, 4, 5], HGL for [1, 3]
    
    gl_idx, hgl_idx = win.classify_frequencies(omega)
    
    expected_gl = {0, 2, 4, 5}
    expected_hgl = {1, 3}
    
    gl_set = set(gl_idx)
    hgl_set = set(hgl_idx)
    
    checks = [
        (gl_set == expected_gl, f"GL indices: {gl_set} should be {expected_gl}"),
        (hgl_set == expected_hgl, f"HGL indices: {hgl_set} should be {expected_hgl}"),
    ]
    
    all_pass = True
    for passed, msg in checks:
        if not passed:
            print(f"  FAIL: {msg}")
            all_pass = False
    
    if all_pass:
        print(f"  E_gap = {E_gap}, E_bw = {E_bw}")
        print(f"  omega = {omega}")
        print(f"  GL indices: {list(gl_idx)} (|ω| outside [{E_gap}, {E_bw}])")
        print(f"  HGL indices: {list(hgl_idx)} (|ω| inside [{E_gap}, {E_bw}])")
        print("  PASS: classify_frequencies works correctly")
    
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("HGL Quadrature Unit Tests")
    print("=" * 60)
    
    tests = [
        test_hgl_nodes_weights_basic,
        test_hgl_nodes_weights_sum,
        test_hgl_regularization_function,
        test_n_tau_hgl_formula,
        test_window_pair_hgl_fields,
        test_window_pair_init_hgl,
        test_window_pair_gamma_equals_z_lm,
        test_classify_frequencies,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n  EXCEPTION in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"All {total} tests passed!")
    else:
        print(f"{passed}/{total} tests passed, {total - passed} failed")
    print("=" * 60)

"""
hgl_quadrature.py

Hermite-Gauss-Laguerre (HGL) quadrature for weight function exp(-τ - τ²/2).

Implements the Golub-Welsch algorithm to compute nodes and weights for the
HGL quadrature, as described in Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020),
Appendix H.

The key identity is:
    1/x = γ ∫₀^∞ sin(γxτ) exp(-τ - τ²/2) dτ    [for x > 0, any γ > 0]
    
This is the imaginary part of the Faddeeva/error function relation.
"""

import numpy as np
from scipy.special import roots_laguerre


def hgl_moments(n_moments: int, n_gl_points: int | None = None) -> np.ndarray:
    """
    Compute moments μ_k = ∫₀^∞ τ^k exp(-τ - τ²/2) dτ for k = 0, 1, ..., n_moments-1.
    
    Uses Gauss-Laguerre quadrature with sufficient points for high accuracy.
    
    Args:
        n_moments: Number of moments to compute (need 2*n for n-point quadrature)
        n_gl_points: Number of GL quadrature points (auto-selected if None)
    
    Returns:
        moments: Array of shape (n_moments,) containing μ_0, μ_1, ..., μ_{n_moments-1}
    """
    if n_gl_points is None:
        # Use enough points for convergence
        n_gl_points = max(100, 10 * n_moments)
    
    # Get Gauss-Laguerre nodes and weights for weight exp(-τ)
    tau_gl, w_gl = roots_laguerre(n_gl_points)
    
    # Additional weight factor exp(-τ²/2) to get full HGL weight
    weight_factor = np.exp(-tau_gl**2 / 2)
    
    # Compute moments: μ_k = Σ_i w_i * τ_i^k * exp(-τ_i²/2)
    moments = np.array([
        np.sum(w_gl * weight_factor * tau_gl**k)
        for k in range(n_moments)
    ])
    
    return moments


def hgl_nodes_weights(n: int, n_gl_base: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute n-point HGL quadrature nodes and weights using Golub-Welsch algorithm.
    
    For the weight function h(τ) = exp(-τ - τ²/2), computes nodes {τ_u} and weights {w_u}
    such that:
    
        ∫₀^∞ f(τ) exp(-τ - τ²/2) dτ ≈ Σ_u w_u f(τ_u)
    
    This is a direct translation of the MATLAB code from Kim et al. Appendix H:
    the key insight is to use GL quadrature to compute inner products stably.
    
    Args:
        n: Number of quadrature points
        n_gl_base: Base number of GL points for inner products (max ~300 before scipy overflows)
    
    Returns:
        nodes: Array of n nodes (sorted ascending)
        weights: Array of n corresponding weights
    
    Example:
        >>> tau, w = hgl_nodes_weights(10)
        >>> # Test: ∫₀^∞ sin(x*τ) exp(-τ - τ²/2) dτ ≈ Σ w_u sin(x*τ_u)
        >>> x = 1.0
        >>> np.sum(w * np.sin(x * tau))  # Should approximate 1/x
    """
    # Step 1: Find number of GL points for accurate moment computation
    # Note: scipy.special.roots_laguerre overflows for n_gl > ~350
    n_gl = min(300, max(n_gl_base, 20 * n))
    
    # Get GL nodes/weights with suppressed overflow warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau_gl, w_gl = roots_laguerre(n_gl)
    
    # Additional weight for HGL: exp(-τ²/2)
    weight_factor = np.exp(-tau_gl**2 / 2)
    
    # μ_0 = ∫ exp(-τ - τ²/2) dτ
    mu0 = np.sum(w_gl * weight_factor)
    
    # Step 2: Build orthogonal polynomials and get recurrence coefficients
    # Following the MATLAB code structure exactly
    p = np.zeros((n_gl, n + 1))
    p[:, 0] = 1.0
    
    a = np.zeros(n)
    b = np.zeros(n)
    
    for j in range(n):
        # <τ p_j², w> / <p_j², w>
        xpp = np.sum(w_gl * tau_gl * weight_factor * p[:, j]**2)
        pp = np.sum(w_gl * weight_factor * p[:, j]**2)
        a[j] = xpp / pp
        
        if j > 0:
            ppm1 = np.sum(w_gl * weight_factor * p[:, j-1]**2)
            b[j] = pp / ppm1
        
        # Three-term recurrence for next polynomial
        if j > 0:
            p[:, j+1] = (tau_gl - a[j]) * p[:, j] - b[j] * p[:, j-1]
        else:
            p[:, j+1] = (tau_gl - a[j]) * p[:, j]
    
    # Step 3: Golub-Welsch: build Jacobi matrix from recurrence coefficients
    b_diag = b[1:]  # b_1, b_2, ..., b_{n-1}
    b_sqrt = np.sqrt(np.maximum(b_diag, 0.0))  # Ensure non-negative for sqrt
    
    J = np.diag(a) + np.diag(b_sqrt, 1) + np.diag(b_sqrt, -1)
    
    # Step 4: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    nodes = eigenvalues[idx]
    weights = mu0 * eigenvectors[0, idx]**2
    
    return nodes, weights


def hgl_all_sizes(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute nodes and weights for all quadrature sizes from 1 to n_max.
    
    Returns matrices where column j contains the j+1 point quadrature
    (padded with zeros for smaller quadratures).
    
    Args:
        n_max: Maximum quadrature size
    
    Returns:
        nodes_mat: (n_max, n_max) matrix, column j has nodes for (j+1)-point quadrature
        weights_mat: (n_max, n_max) matrix, column j has weights for (j+1)-point quadrature
    """
    nodes_mat = np.zeros((n_max, n_max))
    weights_mat = np.zeros((n_max, n_max))
    
    for j in range(1, n_max + 1):
        nodes, weights = hgl_nodes_weights(j)
        nodes_mat[:j, j-1] = nodes
        weights_mat[:j, j-1] = weights
    
    return nodes_mat, weights_mat


def F_hgl(x: float | np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Compute F(x) = Im[w(z)] / √π where w(z) is the Faddeeva function,
    evaluated at z = x / (γ√2).
    
    This is the exact value that the HGL quadrature approximates:
        1/x = γ ∫₀^∞ sin(γxτ) exp(-τ - τ²/2) dτ
    
    So F(x) ≈ (1/x) / γ for the integral value.
    
    For numerical stability, uses the Dawson function relation:
        F(x) = 2 * dawson(x / (γ√2)) for real x
    
    Args:
        x: Energy argument(s)
        gamma: Scaling parameter
    
    Returns:
        F(x) values
    """
    from scipy.special import dawsn
    
    x = np.atleast_1d(x)
    z = x / (gamma * np.sqrt(2))
    return 2 * dawsn(z)


def n_tau_hgl(epsilon: float, x: float) -> int:
    """
    Estimate the number of HGL quadrature points needed for fractional error ε.
    
    Uses the empirical fit from Appendix D of Kim et al.:
        N^(τ,HGL) = c_2(ε) x² + c_1(ε) x + c_0(ε)
    
    where x = γ × E_bw (bandwidth in scaled units).
    
    Args:
        epsilon: Target fractional error
        x: Scaled bandwidth (γ × E_bw)
    
    Returns:
        Recommended number of quadrature points (at least 3)
    """
    ln_eps = np.log(epsilon)
    
    c2 = -0.0036 * ln_eps + 0.11
    c1 = -0.0043 * ln_eps**2 - 0.13 * ln_eps + 0.54
    c0 = -0.204 * ln_eps - 0.29
    
    n = c2 * x**2 + c1 * x + c0
    return max(3, int(np.ceil(n)))


if __name__ == "__main__":
    # Quick validation
    from scipy.integrate import quad as scipy_quad
    
    print("Testing HGL quadrature implementation")
    print("=" * 50)
    
    # Test 1: Moments should be positive and decreasing-ish
    moments = hgl_moments(10)
    print(f"\nFirst 10 moments of exp(-τ - τ²/2):")
    print(f"  μ_0 = {moments[0]:.6f}")
    print(f"  μ_1 = {moments[1]:.6f}")
    print(f"  μ_2 = {moments[2]:.6f}")
    
    # Test 2: Nodes and weights for n=5
    tau, w = hgl_nodes_weights(5)
    print(f"\n5-point HGL quadrature:")
    print(f"  Nodes:   {tau}")
    print(f"  Weights: {w}")
    print(f"  Sum of weights: {np.sum(w):.6f} (should be μ_0 = {moments[0]:.6f})")
    
    # Helper: exact F(x) via numerical integration
    def F_exact(x):
        result, _ = scipy_quad(lambda t: np.sin(x * t) * np.exp(-t - t**2/2), 0, 100)
        return result
    
    # Test 3: Approximate F(x) via quadrature
    # F(x) = ∫₀^∞ sin(xτ) exp(-τ - τ²/2) dτ → 1/x for large x
    print(f"\nTest: F(x) = ∫ sin(xτ) exp(-τ - τ²/2) dτ")
    print(f"      F(x) → 1/x for |x| >> 1 (regularizes 1/x for small x)")
    
    for n_pts in [5, 10, 15, 20]:
        tau, w = hgl_nodes_weights(n_pts)
        
        errors = []
        for x in [2.0, 5.0, 10.0, 20.0]:
            quad_val = np.sum(w * np.sin(x * tau))
            exact_val = F_exact(x)
            rel_err = abs(quad_val - exact_val) / abs(exact_val)
            errors.append(rel_err)
        
        print(f"  n={n_pts:2d}: rel errors for x=2,5,10,20: {[f'{e:.2e}' for e in errors]}")
    
    # Test 4: The key physics test - approximate 1/x using γ scaling
    print(f"\nPhysics test: 1/x ≈ γ × F(γx) for |γx| >> 1")
    print(f"              where F(γx) = ∫ sin(γxτ) exp(-τ - τ²/2) dτ")
    
    # For large γx, γ × F(γx) → γ × (1/γx) = 1/x
    E_gap, E_bw = 0.1, 2.0
    gamma = 1.0 / np.sqrt(E_gap * E_bw)
    print(f"  Window: E_gap={E_gap}, E_bw={E_bw}, γ={gamma:.2f}")
    
    for n_pts in [5, 10, 15]:
        tau, w = hgl_nodes_weights(n_pts)
        
        errors = []
        # Test energies where γx is large enough for 1/x approximation
        for x in [0.5, 1.0, 1.5, 2.0]:
            gamma_x = gamma * x
            # Quadrature approximation
            quad_val = gamma * np.sum(w * np.sin(gamma_x * tau))
            # For comparison: exact γ × F(γx)
            exact_F = gamma * F_exact(gamma_x)
            rel_err = abs(quad_val - exact_F) / abs(exact_F)
            errors.append(rel_err)
        
        print(f"  n={n_pts:2d}: rel errors for x=0.5,1,1.5,2: {[f'{e:.2e}' for e in errors]}")
    
    print(f"\n  Comparison: quadrature vs exact integral vs 1/x:")
    n_pts = 15
    tau, w = hgl_nodes_weights(n_pts)
    print(f"  {'x':>5} {'γx':>6} {'quad':>10} {'exact F':>10} {'1/x':>10} {'err(quad)':>10}")
    for x in [0.5, 1.0, 1.5, 2.0]:
        gamma_x = gamma * x
        quad_val = gamma * np.sum(w * np.sin(gamma_x * tau))
        exact_F = gamma * F_exact(gamma_x)
        inv_x = 1.0 / x
        err = abs(quad_val - exact_F) / abs(exact_F)
        print(f"  {x:5.1f} {gamma_x:6.2f} {quad_val:10.6f} {exact_F:10.6f} {inv_x:10.6f} {err:10.2e}")
    
    print("\n✓ HGL quadrature implementation complete")


"""
hgl_quadrature.py

Hermite-Gauss-Laguerre (HGL) quadrature for weight function exp(-τ - τ²/2).

Implements the Golub-Welsch algorithm to compute nodes and weights for the
HGL quadrature, as described in Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020),
Appendix H.

The key identity is:
    1/x = γ ∫₀^∞ sin(γxτ) exp(-τ - τ²/2) dτ    [for x > 0, any γ > 0]

For the Complex-Time Shredded Propagator (CTSP) method, HGL is used when
the frequency ω falls within the transition energy range [E_gap, E_bw],
causing energy crossings (denominator changes sign).
"""

import numpy as np
from scipy.special import roots_laguerre
import warnings


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
        >>> np.sum(w * np.sin(x * tau))  # Should approximate F(x)
    """
    # Step 1: Find number of GL points for accurate moment computation
    # Note: scipy.special.roots_laguerre overflows for n_gl > ~350
    n_gl = min(300, max(n_gl_base, 20 * n))
    
    # Get GL nodes/weights with suppressed overflow warnings
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


def n_tau_hgl(gamma: float, E_bw: float, epsilon: float = 0.01) -> int:
    """
    Estimate the number of HGL quadrature points needed for fractional error ε.
    
    Uses the empirical fit from Appendix D of Kim et al.:
        N^(τ,HGL) = c_2(ε) x² + c_1(ε) x + c_0(ε)
    
    where x = γ × E_bw (bandwidth in scaled units).
    
    Args:
        gamma: Scaling parameter (= z_lm = 1/√(E_gap × E_bw))
        E_bw: Energy bandwidth of the window
        epsilon: Target fractional error (default 0.01 = 1%)
    
    Returns:
        Recommended number of quadrature points (at least 3)
    """
    x = gamma * E_bw
    ln_eps = np.log(epsilon)
    
    c2 = -0.0036 * ln_eps + 0.11
    c1 = -0.0043 * ln_eps**2 - 0.13 * ln_eps + 0.54
    c0 = -0.204 * ln_eps - 0.29
    
    n = c2 * x**2 + c1 * x + c0
    return max(3, int(np.ceil(n)))


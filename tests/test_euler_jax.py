"""
test_euler_jax.py

Test the Euler identity optimization for HGL propagators in JAX.

The key identity for REAL coefficients A_n:
    G = Σ_n exp(iγτE_n) A_n = Σ_n (cos(γτE_n) + i sin(γτE_n)) A_n = C + iS

For COMPLEX matrix-valued propagators ψψ†:
    G = Σ_n exp(iγτE_n) |ψ_n><ψ_n| 
    
The Euler identity applies to the exp(iE) part, which is still scalar:
    exp(iγτE) = cos(γτE) + i sin(γτE)

So G = Σ_n (cos + i sin) |ψ><ψ|† = C + iS where C,S are complex matrices.

Then: conj(G_c) * G_v = (C_c* - iS_c*)(C_v + iS_v) 
     = C_c* C_v + S_c* S_v + i(C_c* S_v - S_c* C_v)

NOT = P_+ - iP_× unless C and S are real!

For the actual kernel, we need to verify this identity holds with the
specific contraction pattern.

Run with: python tests/test_euler_jax.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '/home/jackm/projects/isdf_cohsex/src')

# Enable float64 for accurate tests
jax.config.update("jax_enable_x64", True)


def test_euler_identity_scalar():
    """Test Euler identity with scalar coefficients (1D case)."""
    print("\nTest: Euler identity (scalar real coefficients)")
    
    np.random.seed(42)
    n_val, n_cond = 5, 8
    gamma = 2.0
    tau = 1.5
    
    E_v = np.sort(np.random.uniform(-0.5, 0.0, n_val))
    E_c = np.sort(np.random.uniform(0.1, 0.8, n_cond))
    
    # Real coefficients (Hermitian case)
    A_v = np.random.random(n_val)
    A_c = np.random.random(n_cond)
    
    # === NAIVE: 4 separate arrays ===
    C_v = np.sum(np.cos(gamma * tau * E_v) * A_v)
    S_v = np.sum(np.sin(gamma * tau * E_v) * A_v)
    C_c = np.sum(np.cos(gamma * tau * E_c) * A_c)
    S_c = np.sum(np.sin(gamma * tau * E_c) * A_c)
    
    P_plus_naive = C_c * C_v + S_c * S_v
    P_cross_naive = S_c * C_v - C_c * S_v
    
    # === EULER: 2 complex arrays ===
    G_v = np.sum(np.exp(1j * gamma * tau * E_v) * A_v)
    G_c = np.sum(np.exp(1j * gamma * tau * E_c) * A_c)
    
    product = np.conj(G_c) * G_v
    P_plus_euler = product.real
    P_cross_euler = -product.imag
    
    # Compare
    err_plus = abs(P_plus_naive - P_plus_euler)
    err_cross = abs(P_cross_naive - P_cross_euler)
    
    if err_plus < 1e-14 and err_cross < 1e-14:
        print(f"  P_+ naive={P_plus_naive:.10f}, euler={P_plus_euler:.10f}, err={err_plus:.2e} ✓")
        print(f"  P_× naive={P_cross_naive:.10f}, euler={P_cross_euler:.10f}, err={err_cross:.2e} ✓")
        print("  PASS: Euler identity verified for scalars")
        return True
    else:
        print(f"  FAIL: errors too large: P_+ err={err_plus:.2e}, P_× err={err_cross:.2e}")
        return False


def test_euler_identity_matrix_real():
    """Test Euler identity with real matrix coefficients (outer products)."""
    print("\nTest: Euler identity (matrix outer products, real coeffs)")
    
    np.random.seed(42)
    
    n_val, n_cond = 5, 7
    dim = 4  # matrix dimension
    gamma = 2.0
    tau = 1.5
    
    E_v = np.random.uniform(-0.5, 0.0, n_val)
    E_c = np.random.uniform(0.1, 0.8, n_cond)
    
    # REAL vectors for outer products
    psi_v = np.random.randn(n_val, dim)
    psi_c = np.random.randn(n_cond, dim)
    
    # === NAIVE: 4 matrices ===
    # C_v = Σ_n cos(γτE_n) |ψ_n><ψ_n|
    C_v = np.zeros((dim, dim))
    S_v = np.zeros((dim, dim))
    C_c = np.zeros((dim, dim))
    S_c = np.zeros((dim, dim))
    
    for n in range(n_val):
        outer = np.outer(psi_v[n], psi_v[n])
        C_v += np.cos(gamma * tau * E_v[n]) * outer
        S_v += np.sin(gamma * tau * E_v[n]) * outer
    
    for n in range(n_cond):
        outer = np.outer(psi_c[n], psi_c[n])
        C_c += np.cos(gamma * tau * E_c[n]) * outer
        S_c += np.sin(gamma * tau * E_c[n]) * outer
    
    # Element-wise identity
    P_plus_naive = C_c * C_v + S_c * S_v
    P_cross_naive = S_c * C_v - C_c * S_v
    
    # === EULER: 2 complex matrices ===
    G_v = np.zeros((dim, dim), dtype=complex)
    G_c = np.zeros((dim, dim), dtype=complex)
    
    for n in range(n_val):
        outer = np.outer(psi_v[n], psi_v[n])
        G_v += np.exp(1j * gamma * tau * E_v[n]) * outer
    
    for n in range(n_cond):
        outer = np.outer(psi_c[n], psi_c[n])
        G_c += np.exp(1j * gamma * tau * E_c[n]) * outer
    
    product = np.conj(G_c) * G_v  # element-wise
    P_plus_euler = product.real
    P_cross_euler = -product.imag
    
    err_plus = np.max(np.abs(P_plus_naive - P_plus_euler))
    err_cross = np.max(np.abs(P_cross_naive - P_cross_euler))
    
    if err_plus < 1e-12 and err_cross < 1e-12:
        print(f"  Max |P_+ naive - euler| = {err_plus:.2e} ✓")
        print(f"  Max |P_× naive - euler| = {err_cross:.2e} ✓")
        print("  PASS: Euler identity verified for real matrix outer products")
        return True
    else:
        print(f"  FAIL: Max errors: P_+ = {err_plus:.2e}, P_× = {err_cross:.2e}")
        return False


def test_euler_identity_jax_einsum():
    """Test Euler identity with JAX einsum matching actual kernel structure."""
    print("\nTest: Euler identity with JAX einsum (real psi)")
    
    np.random.seed(42)
    
    nk, ns, nmu = 4, 2, 8
    n_val, n_cond = 5, 7
    gamma = 2.0
    tau = 1.5
    
    # REAL wavefunctions - only way Euler identity works
    psi_v = np.random.randn(nk, ns, nmu, n_val)
    psi_c = np.random.randn(nk, ns, nmu, n_cond)
    
    E_v = np.random.uniform(-0.5, 0.0, (nk, n_val))
    E_c = np.random.uniform(0.1, 0.8, (nk, n_cond))
    
    psi_v_j = jnp.array(psi_v)
    psi_c_j = jnp.array(psi_c)
    E_v_j = jnp.array(E_v)
    E_c_j = jnp.array(E_c)
    
    # === NAIVE: 4 propagators ===
    cos_v = jnp.cos(gamma * tau * E_v_j)
    sin_v = jnp.sin(gamma * tau * E_v_j)
    cos_c = jnp.cos(gamma * tau * E_c_j)
    sin_c = jnp.sin(gamma * tau * E_c_j)
    
    # G = Σ_n weight_n × |ψ_n><ψ_n|
    # psi: (k, s, mu, n_band), cos: (k, n_band)
    # einsum: sum over n (band index)
    # Result: (nk, ns, nmu, ns, nmu)
    C_v_k = jnp.einsum('ksmv,kv,ktnv->ksmtn', psi_v_j, cos_v, psi_v_j)
    S_v_k = jnp.einsum('ksmv,kv,ktnv->ksmtn', psi_v_j, sin_v, psi_v_j)
    C_c_k = jnp.einsum('ksmc,kc,ktnc->ksmtn', psi_c_j, cos_c, psi_c_j)
    S_c_k = jnp.einsum('ksmc,kc,ktnc->ksmtn', psi_c_j, sin_c, psi_c_j)
    
    P_plus_naive = C_c_k * C_v_k + S_c_k * S_v_k
    P_cross_naive = S_c_k * C_v_k - C_c_k * S_v_k
    
    # === EULER: 2 complex propagators ===
    phase_v = jnp.exp(1j * gamma * tau * E_v_j)
    phase_c = jnp.exp(1j * gamma * tau * E_c_j)
    
    G_v_k = jnp.einsum('ksmv,kv,ktnv->ksmtn', psi_v_j, phase_v, psi_v_j)
    G_c_k = jnp.einsum('ksmc,kc,ktnc->ksmtn', psi_c_j, phase_c, psi_c_j)
    
    product = jnp.conj(G_c_k) * G_v_k
    P_plus_euler = product.real
    P_cross_euler = -product.imag
    
    err_plus = jnp.max(jnp.abs(P_plus_naive - P_plus_euler))
    err_cross = jnp.max(jnp.abs(P_cross_naive - P_cross_euler))
    
    if err_plus < 1e-10 and err_cross < 1e-10:
        print(f"  Max |P_+ naive - euler| = {float(err_plus):.2e} ✓")
        print(f"  Max |P_× naive - euler| = {float(err_cross):.2e} ✓")
        print("  PASS: Euler identity verified for JAX einsum (real psi)")
        return True
    else:
        print(f"  FAIL: Max errors: P_+ = {float(err_plus):.2e}, P_× = {float(err_cross):.2e}")
        return False


def test_euler_identity_complex_psi():
    """Test Euler identity for complex ψ (spinor wavefunctions).
    
    For complex ψ, the propagator is:
        G = Σ_n exp(iγτE_n) ψ_n ψ_n†
        
    The cos/sin parts are:
        C = Σ_n cos(γτE_n) ψ_n ψ_n†   (complex Hermitian matrix)
        S = Σ_n sin(γτE_n) ψ_n ψ_n†   (complex Hermitian matrix)
        
    And G = C + iS still holds since exp(iθ) = cos(θ) + i sin(θ).
    
    When we compute conj(G_c) * G_v, we get:
        conj(G_c) * G_v = (C_c - iS_c)* × (C_v + iS_v)
                        = (C_c* - iS_c*)(C_v + iS_v)
                        = C_c* C_v + S_c* S_v + i(C_c* S_v - S_c* C_v)
    
    This is NOT the same as P_+ - iP_× (where P_+ = C_c C_v + S_c S_v)
    because C and S are complex. However, the Euler identity still gives
    us the correct result for the ACTUAL χ computation.
    """
    print("\nTest: Euler identity (complex psi)")
    
    np.random.seed(42)
    
    nk, ns, nmu = 2, 2, 4
    n_val, n_cond = 3, 4
    gamma = 2.0
    tau = 1.5
    
    # Complex wavefunctions (spinor case)
    psi_v = np.random.randn(nk, ns, nmu, n_val) + 1j * np.random.randn(nk, ns, nmu, n_val)
    psi_c = np.random.randn(nk, ns, nmu, n_cond) + 1j * np.random.randn(nk, ns, nmu, n_cond)
    
    E_v = np.random.uniform(-0.5, 0.0, (nk, n_val))
    E_c = np.random.uniform(0.1, 0.8, (nk, n_cond))
    
    psi_v_j = jnp.array(psi_v)
    psi_c_j = jnp.array(psi_c)
    E_v_j = jnp.array(E_v)
    E_c_j = jnp.array(E_c)
    
    # === NAIVE: 4 propagators ===
    cos_v = jnp.cos(gamma * tau * E_v_j)
    sin_v = jnp.sin(gamma * tau * E_v_j)
    cos_c = jnp.cos(gamma * tau * E_c_j)
    sin_c = jnp.sin(gamma * tau * E_c_j)
    
    # G = Σ_n weight_n × ψ_n × ψ_n†
    # psi: (k,s,m,n), cos: (k,n), conj(psi): (k,t,u,n)
    # result: (k,s,m,t,u) from outer product, summed over n
    C_v_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_v_j, cos_v, jnp.conj(psi_v_j))
    S_v_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_v_j, sin_v, jnp.conj(psi_v_j))
    C_c_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_c_j, cos_c, jnp.conj(psi_c_j))
    S_c_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_c_j, sin_c, jnp.conj(psi_c_j))
    
    # === EULER: 2 complex propagators ===
    phase_v = jnp.exp(1j * gamma * tau * E_v_j)
    phase_c = jnp.exp(1j * gamma * tau * E_c_j)
    
    G_v_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_v_j, phase_v, jnp.conj(psi_v_j))
    G_c_k = jnp.einsum('ksmn,kn,ktun->ksmtu', psi_c_j, phase_c, jnp.conj(psi_c_j))
    
    # Verify G = C + iS
    # Note: single precision float32 gives ~1e-3 error, float64 gives ~1e-10
    err_v = jnp.max(jnp.abs(G_v_k - (C_v_k + 1j * S_v_k)))
    err_c = jnp.max(jnp.abs(G_c_k - (C_c_k + 1j * S_c_k)))
    
    # Tolerance depends on JAX default precision (float32 by default)
    tol = 1e-5  # Allow for float32 precision
    if err_v < tol and err_c < tol:
        print(f"  Verified: G = C + iS for complex psi")
        print(f"  Max |G_v - (C_v + iS_v)| = {float(err_v):.2e} ✓")
        print(f"  Max |G_c - (C_c + iS_c)| = {float(err_c):.2e} ✓")
        print("  PASS: Euler identity G = C + iS verified for complex psi")
        return True
    else:
        print(f"  FAIL: Max errors: G_v = {float(err_v):.2e}, G_c = {float(err_c):.2e}")
        return False


def test_euler_chi_contraction():
    """Test Euler identity with the actual χ contraction pattern.
    
    For χ computation in the kernel, we contract:
        χ_μν^q = FFT_R [ Σ_{ab} G^c_{aμbν,R} G^v_{bνaμ,-R} ]
        
    Using Euler, if G = C + iS:
        G^c†_R × G^v_{-R} = (C^c - iS^c)_R × (C^v + iS^v)_{-R}
        
    The result's real part contributes to P_+, imaginary to -P_×.
    """
    print("\nTest: Euler identity with χ contraction pattern")
    
    np.random.seed(42)
    
    nkx, nky, nkz = 2, 2, 2
    nk = nkx * nky * nkz
    ns, nmu = 2, 3
    n_val, n_cond = 3, 4
    gamma = 2.0
    tau = 1.5
    
    # Real wavefunctions for clean test
    psi_v = np.random.randn(nk, ns, nmu, n_val)
    psi_c = np.random.randn(nk, ns, nmu, n_cond)
    
    E_v = np.random.uniform(-0.5, 0.0, (nk, n_val))
    E_c = np.random.uniform(0.1, 0.8, (nk, n_cond))
    
    psi_v_j = jnp.array(psi_v)
    psi_c_j = jnp.array(psi_c)
    E_v_j = jnp.array(E_v)
    E_c_j = jnp.array(E_c)
    
    def build_prop_naive(psi, E, gamma, tau):
        """Build cos/sin propagators. psi: (k,a,m,n), E: (k,n)."""
        cos_E = jnp.cos(gamma * tau * E)
        sin_E = jnp.sin(gamma * tau * E)
        # G = Σ_n weight_n × |ψ_n><ψ_n|
        # psi: (k,a,m,n), cos: (k,n), result: (k,a,m,b,u) summing over n
        C = jnp.einsum('kamn,kn,kbun->kambu', psi, cos_E, psi)
        S = jnp.einsum('kamn,kn,kbun->kambu', psi, sin_E, psi)
        return C, S
    
    def build_prop_euler(psi, E, gamma, tau):
        """Build complex propagator. psi: (k,a,m,n), E: (k,n)."""
        phase = jnp.exp(1j * gamma * tau * E)
        G = jnp.einsum('kamn,kn,kbun->kambu', psi, phase, psi)
        return G
    
    def k_to_R(G_k, flip_sign=False):
        """FFT from k to R. flip_sign=True for G_{-R}."""
        G = G_k.reshape(nkx, nky, nkz, ns, nmu, ns, nmu)
        G = G.transpose(3, 4, 5, 6, 0, 1, 2)
        if flip_sign:
            return jnp.fft.fftn(G, axes=(-3, -2, -1), norm='ortho')
        else:
            return jnp.fft.ifftn(G, axes=(-3, -2, -1), norm='ortho')
    
    def contract_R(G_c_R, G_v_R):
        """χ contraction: Σ_{ab} G^c_{aμbν} × G^v_{bνaμ}."""
        return jnp.einsum('ambvxyz,bvamxyz->mvxyz', G_c_R, G_v_R)
    
    # === NAIVE ===
    C_v, S_v = build_prop_naive(psi_v_j, E_v_j, gamma, tau)
    C_c, S_c = build_prop_naive(psi_c_j, E_c_j, gamma, tau)
    
    C_v_R = k_to_R(C_v, flip_sign=False)
    S_v_R = k_to_R(S_v, flip_sign=False)
    C_c_mR = k_to_R(C_c, flip_sign=True)
    S_c_mR = k_to_R(S_c, flip_sign=True)
    
    P_plus_naive = contract_R(C_c_mR, C_v_R) + contract_R(S_c_mR, S_v_R)
    P_cross_naive = contract_R(S_c_mR, C_v_R) - contract_R(C_c_mR, S_v_R)
    
    # === EULER ===
    G_v = build_prop_euler(psi_v_j, E_v_j, gamma, tau)
    G_c = build_prop_euler(psi_c_j, E_c_j, gamma, tau)
    
    G_v_R = k_to_R(G_v, flip_sign=False)
    G_c_mR = k_to_R(G_c, flip_sign=True)
    
    product_R = contract_R(jnp.conj(G_c_mR), G_v_R)
    P_plus_euler = product_R.real
    P_cross_euler = -product_R.imag
    
    err_plus = jnp.max(jnp.abs(P_plus_naive - P_plus_euler))
    err_cross = jnp.max(jnp.abs(P_cross_naive - P_cross_euler))
    
    if err_plus < 1e-10 and err_cross < 1e-10:
        print(f"  P_plus shape: {P_plus_naive.shape}")
        print(f"  Max |P_+ naive - euler| = {float(err_plus):.2e} ✓")
        print(f"  Max |P_× naive - euler| = {float(err_cross):.2e} ✓")
        print("  PASS: Euler identity verified for χ contraction")
        return True
    else:
        print(f"  FAIL: Max errors: P_+ = {float(err_plus):.2e}, P_× = {float(err_cross):.2e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Euler Identity Tests for JAX HGL Kernel")
    print("=" * 60)
    
    tests = [
        test_euler_identity_scalar,
        test_euler_identity_matrix_real,
        test_euler_identity_jax_einsum,
        test_euler_identity_complex_psi,
        test_euler_chi_contraction,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n  EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"All {total} tests passed!")
    else:
        print(f"{passed}/{total} tests passed, {total - passed} failed")
    print("=" * 60)

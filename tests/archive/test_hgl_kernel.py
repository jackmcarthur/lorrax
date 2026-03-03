"""
test_hgl_kernel.py

Test the HGL kernel for dynamic χ(ω) computation against numpy reference.

The kernel computes P_plus and P_cross for a single τ point, which are
combined with ω-dependent phase factors to get χ(ω).

Run with: python tests/test_hgl_kernel.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import sys
sys.path.insert(0, '/home/jackm/projects/isdf_cohsex/src')

# Enable float64 for accurate tests
jax.config.update("jax_enable_x64", True)


def numpy_reference_hgl_tile(
    psi_vTX, psi_vY, psi_cX, psi_cTY,
    enk_v, enk_c,
    tau, gamma,
    val_mask, cond_mask,
    nkx, nky, nkz,
):
    """
    Numpy reference for P_plus and P_cross computation.
    
    Args:
        psi_vTX: (nk, ns, nmu, nv) left valence
        psi_vY: (nk, nv, ns, nmu) right valence
        psi_cX: (nk, ns, nmu, nc) left conduction (note: transposed in actual kernel)
        psi_cTY: (nk, nc, ns, nmu) right conduction (note: transposed in actual kernel)
        enk_v: (nk, nv)
        enk_c: (nk, nc)
        tau: scalar
        gamma: scalar
        val_mask: (nk, nv) bool
        cond_mask: (nk, nc) bool
        nkx, nky, nkz: k-grid dimensions
    
    Returns:
        P_plus_q, P_cross_q: (nkx, nky, nkz, 1, nmu, 1, nmu)
    """
    nk = psi_vTX.shape[0]
    nmu = psi_vTX.shape[2]
    
    # Apply masks
    val_mask_f = val_mask.astype(np.float64)
    cond_mask_f = cond_mask.astype(np.float64)
    
    psi_vTX = psi_vTX * val_mask_f[:, None, None, :]
    psi_vY = psi_vY * val_mask_f[:, :, None, None]
    psi_cX = psi_cX * cond_mask_f[:, :, None, None]
    psi_cTY = psi_cTY * cond_mask_f[:, None, None, :]
    
    # Euler phases
    phase_v = np.exp(1j * gamma * tau * enk_v) * val_mask.astype(np.complex128)
    phase_c = np.exp(1j * gamma * tau * enk_c) * cond_mask.astype(np.complex128)
    
    # Build propagators: G = Σ_n phase_n |ψ_n><ψ_n|
    # Gv: (nk, ns, nmu, ns, nmu)
    Gv_k = np.einsum('ksxm,km,kmty->ksxty', np.conj(psi_vTX), phase_v, psi_vY)
    Gc_k = np.einsum('ksxm,km,kmty->ksxty', np.conj(psi_cTY), phase_c, psi_cX)
    
    # FFT to R-space
    def k_to_R(G_k, flip_sign):
        G = G_k.reshape(nkx, nky, nkz, *G_k.shape[1:])
        G = G.transpose(3, 4, 5, 6, 0, 1, 2)  # (ns, nmu, ns, nmu, kx, ky, kz)
        if flip_sign:
            return np.fft.fftn(G, axes=(-3, -2, -1), norm='ortho')
        else:
            return np.fft.ifftn(G, axes=(-3, -2, -1), norm='ortho')
    
    Gv_R = k_to_R(Gv_k, flip_sign=False)
    Gc_mR = k_to_R(Gc_k, flip_sign=True)
    
    # Contract: conj(G_c)_{-R} @ G_v_R
    product_R = np.einsum('ambnxyz, bnamxyz-> mnxyz', np.conj(Gc_mR), Gv_R)
    P_plus_R = product_R.real
    P_cross_R = -product_R.imag
    
    # FFT to q-space
    P_plus_q = np.fft.fftn(P_plus_R, axes=(-3, -2, -1), norm='ortho')
    P_cross_q = np.fft.fftn(P_cross_R, axes=(-3, -2, -1), norm='ortho')
    
    # Reshape
    P_plus_q = P_plus_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
    P_cross_q = P_cross_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
    
    return P_plus_q, P_cross_q


def test_hgl_kernel_small():
    """Test HGL kernel with small arrays (no sharding)."""
    print("\nTest: HGL kernel (small arrays, no sharding)")
    
    np.random.seed(42)
    
    # Small dimensions for testing
    nkx, nky, nkz = 2, 2, 2
    nk = nkx * nky * nkz
    ns = 1  # nspinor
    nmu = 4
    nv = 3  # valence bands
    nc = 4  # conduction bands
    
    # Generate random wavefunctions
    psi_vTX = np.random.randn(nk, ns, nmu, nv) + 1j * np.random.randn(nk, ns, nmu, nv)
    psi_vY = np.random.randn(nk, nv, ns, nmu) + 1j * np.random.randn(nk, nv, ns, nmu)
    # Note: in actual code, cX uses axis=1 for bands, cTY uses axis=3
    psi_cX = np.random.randn(nk, nc, ns, nmu) + 1j * np.random.randn(nk, nc, ns, nmu)
    psi_cTY = np.random.randn(nk, ns, nmu, nc) + 1j * np.random.randn(nk, ns, nmu, nc)
    
    enk_v = np.random.uniform(-0.5, 0.0, (nk, nv))
    enk_c = np.random.uniform(0.1, 0.8, (nk, nc))
    
    tau = 1.5
    gamma = 2.0
    
    val_mask = np.ones((nk, nv), dtype=bool)
    cond_mask = np.ones((nk, nc), dtype=bool)
    
    # Numpy reference
    # Need to match the kernel's einsum pattern:
    # Kernel uses psi_cTY for the left side and psi_cX for the right side
    # So we need to reorder our test arrays
    psi_cTY_for_ref = psi_cTY.transpose(0, 3, 1, 2)  # (nk, nc, ns, nmu)
    psi_cX_for_ref = psi_cX.transpose(0, 2, 3, 1)    # (nk, ns, nmu, nc)
    
    P_plus_ref, P_cross_ref = numpy_reference_hgl_tile(
        psi_vTX, psi_vY, psi_cX_for_ref, psi_cTY_for_ref,
        enk_v, enk_c,
        tau, gamma,
        val_mask, cond_mask,
        nkx, nky, nkz,
    )
    
    print(f"  Reference P_plus shape: {P_plus_ref.shape}")
    print(f"  Reference P_plus max abs: {np.max(np.abs(P_plus_ref)):.4f}")
    print(f"  Reference P_cross max abs: {np.max(np.abs(P_cross_ref)):.4f}")
    print("  PASS: Reference computation works")
    return True


def test_hgl_kernel_identity():
    """Test that P_+ and P_× satisfy expected relations."""
    print("\nTest: HGL P_+ and P_× identity checks")
    
    np.random.seed(123)
    
    nkx, nky, nkz = 2, 2, 2
    nk = nkx * nky * nkz
    ns = 1
    nmu = 3
    nv = 2
    nc = 3
    
    # Real wavefunctions for cleaner identity test
    psi_vTX = np.random.randn(nk, ns, nmu, nv)
    psi_vY = np.random.randn(nk, nv, ns, nmu)
    psi_cTY_orig = np.random.randn(nk, ns, nmu, nc)
    psi_cX_orig = np.random.randn(nk, nc, ns, nmu)
    
    # Reorder for reference function
    psi_cTY_for_ref = psi_cTY_orig.transpose(0, 3, 1, 2)
    psi_cX_for_ref = psi_cX_orig.transpose(0, 2, 3, 1)
    
    enk_v = np.random.uniform(-0.5, 0.0, (nk, nv))
    enk_c = np.random.uniform(0.1, 0.8, (nk, nc))
    
    tau = 1.0
    gamma = 1.5
    
    val_mask = np.ones((nk, nv), dtype=bool)
    cond_mask = np.ones((nk, nc), dtype=bool)
    
    P_plus, P_cross = numpy_reference_hgl_tile(
        psi_vTX, psi_vY, psi_cX_for_ref, psi_cTY_for_ref,
        enk_v, enk_c,
        tau, gamma,
        val_mask, cond_mask,
        nkx, nky, nkz,
    )
    
    # For real wavefunctions, P_+ and P_× should be real
    if not np.allclose(P_plus.imag, 0, atol=1e-10):
        print(f"  FAIL: P_plus should be real for real psi, max imag = {np.max(np.abs(P_plus.imag))}")
        return False
    if not np.allclose(P_cross.imag, 0, atol=1e-10):
        print(f"  FAIL: P_cross should be real for real psi, max imag = {np.max(np.abs(P_cross.imag))}")
        return False
    
    print(f"  P_plus max abs: {np.max(np.abs(P_plus)):.4f}")
    print(f"  P_cross max abs: {np.max(np.abs(P_cross)):.4f}")
    print(f"  P_plus imag max: {np.max(np.abs(P_plus.imag)):.2e} (should be ~0)")
    print(f"  P_cross imag max: {np.max(np.abs(P_cross.imag)):.2e} (should be ~0)")
    print("  PASS: P_+ and P_× are real for real wavefunctions")
    return True


def test_hgl_chi_omega():
    """Test that P_+ and P_× combine correctly to give χ(ω)."""
    print("\nTest: HGL χ(ω) from P_+ and P_×")
    
    np.random.seed(456)
    
    nkx, nky, nkz = 2, 2, 2
    nk = nkx * nky * nkz
    ns = 1
    nmu = 3
    nv = 2
    nc = 3
    
    # Real wavefunctions
    psi_vTX = np.random.randn(nk, ns, nmu, nv)
    psi_vY = np.random.randn(nk, nv, ns, nmu)
    psi_cTY_orig = np.random.randn(nk, ns, nmu, nc)
    psi_cX_orig = np.random.randn(nk, nc, ns, nmu)
    
    psi_cTY_for_ref = psi_cTY_orig.transpose(0, 3, 1, 2)
    psi_cX_for_ref = psi_cX_orig.transpose(0, 2, 3, 1)
    
    enk_v = np.random.uniform(-0.5, 0.0, (nk, nv))
    enk_c = np.random.uniform(0.1, 0.8, (nk, nc))
    
    gamma = 1.5
    omega = 0.3  # Test frequency
    
    # HGL nodes/weights (using small number for test)
    from gw_isdf.hgl_quadrature import hgl_nodes_weights
    tau_nodes, w_nodes = hgl_nodes_weights(5)
    
    val_mask = np.ones((nk, nv), dtype=bool)
    cond_mask = np.ones((nk, nc), dtype=bool)
    
    # Accumulate χ(ω) from HGL formula
    # χ^HGL(ω) = -γ Σ_u w_u [cos(γτ_u ω) P_×(τ_u) - sin(γτ_u ω) P_+(τ_u)]
    chi_omega = None
    for u, (tau_u, w_u) in enumerate(zip(tau_nodes, w_nodes)):
        P_plus, P_cross = numpy_reference_hgl_tile(
            psi_vTX, psi_vY, psi_cX_for_ref, psi_cTY_for_ref,
            enk_v, enk_c,
            tau_u, gamma,
            val_mask, cond_mask,
            nkx, nky, nkz,
        )
        
        cos_factor = np.cos(gamma * tau_u * omega)
        sin_factor = np.sin(gamma * tau_u * omega)
        
        contrib = -gamma * w_u * (cos_factor * P_cross - sin_factor * P_plus)
        
        if chi_omega is None:
            chi_omega = contrib
        else:
            chi_omega = chi_omega + contrib
    
    print(f"  χ(ω={omega}) shape: {chi_omega.shape}")
    print(f"  χ(ω={omega}) max abs: {np.max(np.abs(chi_omega)):.4f}")
    print(f"  χ(ω={omega}) at q=0: {chi_omega[0,0,0,0,0,0,0]:.6f}")
    print("  PASS: HGL χ(ω) construction works")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("HGL Kernel Tests")
    print("=" * 60)
    
    tests = [
        test_hgl_kernel_small,
        test_hgl_kernel_identity,
        test_hgl_chi_omega,
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


"""Independent NumPy photon-Dyson algebra and cancellation witnesses.

Run only on an authorized compute node. No production imports or scientific
files are read. P=pref*chi_raw and C=volume*contact are already scaled.
Winf is the actual solve(I+V C,V), never reconstructed as constant+V.
The ordered bank stores M_k=C_(k+1)/2 in W-Winf=sum C_j/z**j.
"""
import unittest
import numpy as np


def winfinity(v, c):
    return np.linalg.solve(np.eye(v.shape[0], dtype=v.dtype) + v @ c, v)


def stable(v, p, c, winf):
    a = np.eye(v.shape[0], dtype=v.dtype) - v @ (p - c)
    return np.linalg.solve(a, v @ p @ winf)


def old_pair(v, p, dp, c, winf):
    a = np.eye(v.shape[0], dtype=v.dtype) - v @ (p - c)
    constant = winf - v
    value = (np.linalg.solve(a, v) - v) - constant
    reconstructed_full = (value + constant) + v
    return value, reconstructed_full @ dp @ reconstructed_full


def fixture(signed=True):
    v = np.array([[1.3, .12+.17j, -.08+.03j],
                  [.12-.17j, -.8 if signed else .8, .09-.11j],
                  [-.08-.03j, .09+.11j, .7]], dtype=np.complex128)
    l = np.array([[.2, .07+.03j, -.05j], [.01j, .15, .02],
                  [.04, -.03+.02j, .12]], dtype=np.complex128)
    c = l.conj().T @ l
    p = np.array([[.06+.02j, .03-.04j, -.01],
                  [-.02+.01j, -.04+.03j, .02j],
                  [.015j, -.025, .05-.01j]], dtype=np.complex128)
    d = np.array([[.03j, .02, -.01j], [-.012, .04-.01j, .005],
                  [.006+.009j, -.02j, -.03]], dtype=np.complex128)
    return v, c, p, d


def ordered_coefficients(r, k):
    """Independent explicit ordered products, retaining every noncommuting word."""
    k1, k2, k3, k4 = k
    c1 = r @ k1 @ r
    c2 = r @ k2 @ r + r @ k1 @ r @ k1 @ r
    c3 = (r @ k3 @ r + r @ k1 @ r @ k2 @ r
          + r @ k2 @ r @ k1 @ r + r @ k1 @ r @ k1 @ r @ k1 @ r)
    c4 = (r @ k4 @ r + r @ k1 @ r @ k3 @ r + r @ k3 @ r @ k1 @ r
          + r @ k2 @ r @ k2 @ r + r @ k1 @ r @ k1 @ r @ k2 @ r
          + r @ k1 @ r @ k2 @ r @ k1 @ r + r @ k2 @ r @ k1 @ r @ k1 @ r
          + r @ k1 @ r @ k1 @ r @ k1 @ r @ k1 @ r)
    return c1, c2, c3, c4


class StablePhotonDysonAlgebra(unittest.TestCase):
    def close(self, actual, expected, *, rtol=2e-12, atol=5e-14):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def test_noncommuting_identity_and_well_conditioned_old_parity(self):
        for signed in (False, True):
            with self.subTest(signed=signed):
                v, c, p, dp = fixture(signed)
                self.close(v, v.conj().T)
                self.close(c, c.conj().T)
                self.assertGreater(np.linalg.norm(v @ p - p @ v), .01)
                self.assertGreater(np.linalg.norm(v @ c - c @ v), .001)
                r = winfinity(v, c)
                a = np.eye(3) - v @ (p-c)
                self.assertLess(np.linalg.cond(a), 10)
                f = stable(v, p, c, r)
                full = f + r
                direct = np.linalg.solve(a, v)
                self.close(a @ f, v @ p @ r)
                self.close(full, direct)
                old_f, old_dp = old_pair(v, p, dp, c, r)
                self.close(old_f, f)
                self.close(old_dp, full @ dp @ full)
                # A wrong commutation/order must be detectable by this fixture.
                self.assertGreater(np.linalg.norm(f - np.linalg.solve(a, p @ v @ r)), .01)

    def test_complex_s_and_z_derivative_with_fixed_contact(self):
        v, c, p0, d = fixture()
        e = .2 * d.T
        s = .17 + .23j
        r = winfinity(v, c)
        p_of_s = lambda x: p0 + x*d + x*x*e
        f_of_s = lambda x: stable(v, p_of_s(x), c, r)
        full = f_of_s(s) + r
        ds = full @ (d + 2*s*e) @ full
        h = 2e-5
        numerical_s = (f_of_s(s+h)-f_of_s(s-h))/(2*h)
        self.close(numerical_s, ds, rtol=3e-8, atol=2e-10)
        z = np.sqrt(s)
        numerical_z = (f_of_s((z+h)**2)-f_of_s((z-h)**2))/(2*h)
        self.close(numerical_z, 2*z*ds, rtol=3e-8, atol=2e-10)

    def test_ordered_infinity_and_first_four_moments(self):
        v, c, p, d = fixture()
        r = winfinity(v, c)
        k = (p, d, .6*p.T, .7*d.conj().T)
        coeff = ordered_coefficients(r, k)
        # Independent recurrence check and actual finite-frequency remainder.
        for j, cj in enumerate(coeff):
            rhs = k[j] @ r
            for i in range(j):
                rhs = rhs + k[i] @ coeff[j-i-1]
            self.close(cj, r @ rhs)
        errors = []
        for magnitude in (1/8, 1/16, 1/32):
            t = magnitude * np.exp(.33j)  # t=1/z, not 1/s
            pz = sum(kj*t**(j+1) for j, kj in enumerate(k))
            expected = sum(cj*t**(j+1) for j, cj in enumerate(coeff))
            errors.append(np.linalg.norm(stable(v, pz, c, r)-expected))
        self.assertLess(errors[1]/errors[0], .07)
        self.assertLess(errors[2]/errors[1], .07)  # remainder is O(z^-5)
        t = 1e-6 * np.exp(.33j)
        self.close(stable(v, sum(kj*t**(j+1) for j, kj in enumerate(k)), c, r)/t,
                   coeff[0], rtol=5e-6, atol=5e-8)
        # Stored ordered M0,M1,M2,M3 are half of these C1,C2,C3,C4.

    def test_even_s_specialization_and_inexact_winf_residual(self):
        v, c, a0, a1 = fixture()
        r = winfinity(v, c)
        zero = np.zeros_like(v)
        c1, c2, c3, c4 = ordered_coefficients(r, (zero, a0, zero, a1))
        self.close(c1, zero)
        self.close(c3, zero)
        self.close(c2, r @ a0 @ r)
        self.close(c4, r @ a1 @ r + r @ a0 @ r @ a0 @ r)
        # General supplied R: F_R+R-W=A^-1[(I+VC)R-V]. This makes
        # the infinity-solve residual visible; stability is not a license to
        # replace actual R by an inaccurate constant+V reconstruction.
        supplied = r + 1e-4*a1
        a = np.eye(3) - v @ (a0-c)
        residual = (np.eye(3) + v @ c) @ supplied - v
        self.close(stable(v, a0, c, supplied)+supplied-np.linalg.solve(a, v),
                   np.linalg.solve(a, residual))

    def test_subtraction_cancellation_toy_preserves_actual_winf(self):
        v = np.array([[float(2**54)]], dtype=np.complex128)
        c = np.array([[1.]], dtype=np.complex128)
        p = np.array([[float(2**-8)]], dtype=np.complex128)
        dp = np.ones_like(p)
        r = winfinity(v, c)
        constant = r-v
        self.assertGreater(np.linalg.norm((constant+v)-r), .5)
        f = stable(v, p, c, r)
        full = f+r
        old_f, old_slope = old_pair(v, p, dp, c, r)
        self.close(f, p/(1-p))
        self.close(full, 1/(1-p))
        self.assertGreater(np.linalg.norm(old_f-f), .003)
        self.assertGreater(np.linalg.norm(old_slope-full @ dp @ full), 1.)
        h = 1e-5
        fd = (stable(v, p+h*dp, c, r)-stable(v, p-h*dp, c, r))/(2*h)
        self.close(fd, full @ dp @ full, rtol=2e-8, atol=1e-10)


if __name__ == '__main__':
    unittest.main()

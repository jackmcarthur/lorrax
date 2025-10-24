#!/usr/bin/env python3
"""
Numerical check of the nonlocal commutator i[r, V_NL] using synthetic
projectors in QE real-harmonic convention.

We build a minimal model for a single-ℓ channel and one atom at tau=0:
  - Build Y'(K) and ∇_ΩY'(K) via qe_real_sph_harmonics_with_grad.
  - Choose a smooth radial F(q) and its derivative F'(q) (Gaussian model),
    so β(K) = const · F(|K|) Y'(K_hat).
  - Assemble Z(K) rows (R x nG) and dZ(K) (3 x R x nG).
  - Define E (2x2xR x R) spin-diagonal identity.
  - Draw random wavefunction C(band, spinor, G).

Then we compare the analytic commutator assembly:
  i [ conj(dZ) E Z  +  conj(Z) E dZ ] contracted with C
with a finite-difference evaluation of (∇_K + ∇_{K'}) V(K, K')
along Cartesian directions, at the matrix-element level.

Exit code is nonzero if the max relative error exceeds tolerance.
"""

import sys
import math
import argparse
import numpy as np

from isdf.psp.build_projectors_qe import qe_real_sph_harmonics_with_grad


def gaussian_radial(q: np.ndarray, q0: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth radial model F(q) = exp(-(q/q0)^2); returns (F, F')."""
    x = q / max(q0, 1e-12)
    F = np.exp(-(x * x))
    Fp = -2.0 * (x / max(q0, 1e-12)) * F
    return F, Fp


def build_beta_and_grad(l: int, K: np.ndarray, q0: float, pref: complex) -> tuple[np.ndarray, np.ndarray]:
    """Return (Z_rows, dZ_rows):
      - Z_rows: (R, nG)
      - dZ_rows: (3, R, nG)
    with R = 2l+1 (single radial projector) at tau=0, spinless.
    """
    K = np.asarray(K, dtype=float)
    nG = K.shape[0]
    # Angular pieces
    Y, gradY = qe_real_sph_harmonics_with_grad(l, K)
    # Radial pieces
    q = np.linalg.norm(K, axis=1)
    F, Fp = gaussian_radial(q, q0)
    # e_r
    q_safe = np.maximum(q, 1e-12)
    er = (K / q_safe[:, None]).T  # (3, nG)

    # β = pref * F * Y
    Z = (pref * (F[None, :] * Y))  # (R, nG)
    # ∇β = pref * [F' Y e_r + (F/q) ∇ΩY]
    dZ = (pref * (Fp[None, None, :] * Y[None, :, :] * er[:, None, :]
                  + (F / q_safe)[None, None, :] * gradY))  # (3, R, nG)
    return Z.astype(np.complex128), dZ.astype(np.complex128)


def assemble_commutator(Z: np.ndarray, dZ: np.ndarray, C: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Return vNL (3, nb, nb) from analytic assembly for one K-set.
    Z: (R,nG), dZ:(3,R,nG), C:(nb,ns,nG), E:(2,2,R,R)
    """
    nb, ns, nG = C.shape
    P = np.einsum('rg, bsg -> rsb', np.conj(Z), C, optimize=True)       # (R,ns,nb)
    dP = np.einsum('irg, bsg -> irsb', np.conj(dZ), C, optimize=True)   # (3,R,ns,nb)
    term1 = np.einsum('irsn, strq, qtm -> inm', np.conj(dP), E, P, optimize=True)
    term2 = np.einsum('rsn, strq, iqtm -> inm', np.conj(P), E, dP, optimize=True)
    # In Ry a.u., v^NL = - (∇_K + ∇_{K'}) V_NL
    return - (term1 + term2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--l', type=int, default=2)
    p.add_argument('--nG', type=int, default=200)
    p.add_argument('--nb', type=int, default=6)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--q0', type=float, default=0.8)
    p.add_argument('--h', type=float, default=1e-6, help='finite-diff step for K')
    p.add_argument('--tol', type=float, default=5e-6)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    # Random K away from q=0 to avoid singularities
    K = rng.normal(size=(args.nG, 3))
    K = K / np.linalg.norm(K, axis=1, keepdims=True)
    radii = rng.uniform(0.4, 1.5, size=(args.nG, 1))
    K *= radii

    # Single-radial projector per m-slot, spin-diagonal identity E
    R = 2 * args.l + 1
    E = np.zeros((2, 2, R, R), dtype=np.complex128)
    for s in range(2):
        E[s, s] = np.eye(R, dtype=np.complex128)

    # Random C(nb, ns, nG)
    C = rng.normal(size=(args.nb, 2, args.nG)) + 1j * rng.normal(size=(args.nb, 2, args.nG))

    pref = 1.0 + 0.0j  # constants cancel in relative error
    Z, dZ = build_beta_and_grad(args.l, K, args.q0, pref)
    v_analytic = assemble_commutator(Z, dZ, C, E)

    # Numeric: directional derivatives along x,y,z using central differences
    errs = []
    for i in range(3):
        d = np.zeros_like(K)
        d[:, i] = args.h
        Zp, _ = build_beta_and_grad(args.l, K + d, args.q0, pref)
        Zm, _ = build_beta_and_grad(args.l, K - d, args.q0, pref)
        dZ_num = (Zp - Zm) / (2 * args.h)  # (R,nG), this is the i-th directional derivative of Z
        # Form commutator contribution for this direction only: i[ conj(dZ_i) E Z + conj(Z) E dZ_i ] → (nb,nb)
        P = np.einsum('rg, bsg -> rsb', np.conj(Z), C, optimize=True)
        dP_num = np.einsum('rg, bsg -> rsb', np.conj(dZ_num), C, optimize=True)  # (R,ns,nb)
        term1_i = np.einsum('rsn, strq, qtm -> nm', np.conj(dP_num), E, P, optimize=True)
        term2_i = np.einsum('rsn, strq, qtm -> nm', np.conj(P), E, dP_num, optimize=True)
        v_num = - (term1_i + term2_i)
        # Compare analytic vs numeric component-wise
        num = np.linalg.norm(v_num)
        den = np.maximum(1e-12, np.linalg.norm(v_analytic[i]))
        err = float(np.linalg.norm(v_analytic[i] - v_num) / den)
        errs.append(err)
        print(f'dir {i}: rel error {err:.3e}  (||num||={num:.3e}, ||ana||={den:.3e})')

    ok = all(e < args.tol for e in errs)
    if not ok:
        print('Comm utator gradient check FAILED')
        return 2
    print('Commutator gradient check PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

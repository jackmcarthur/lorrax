"""Dense FEAST sanity check (numpy only).

This script validates the FEAST filtering + Rayleigh-Ritz pipeline on a small
dense Hermitian matrix with a controlled spectrum. It is intended as a
debugging tool to verify the equations independent of the JAX matvec path.

Supports FEAST iteration (--feast-iter) and noisy linear solves (--solve-noise)
to simulate GMRES tolerance effects.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    name: str
    a_eV: float
    b_eV: float


def feast_ellipse_quadrature(window: WindowSpec, n_quad: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    a = window.a_eV
    b = window.b_eV
    c = 0.5 * (a + b)
    r_x = 0.5 * (b - a)
    r_y = gamma * r_x
    thetas = np.array([math.pi * (2 * j - 1) / (2 * n_quad) for j in range(1, n_quad + 1)])
    z = c + r_x * np.cos(thetas) + 1j * r_y * np.sin(thetas)
    w = (1.0 / (2.0 * n_quad)) * (r_y * np.cos(thetas) + 1j * r_x * np.sin(thetas))
    return z.astype(np.complex128), w.astype(np.complex128)


def make_test_matrix(n: int, seed: int, e_min: float, e_split: float, e_max: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_low = max(1, n // 5)
    n_high = n - n_low
    low = rng.uniform(e_min, e_split, size=n_low)
    high = rng.uniform(e_split, e_max, size=n_high)
    eigvals = np.sort(np.concatenate([low, high]))

    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    H = q @ np.diag(eigvals) @ q.T
    H = 0.5 * (H + H.T)
    return H, eigvals


def svd_rayleigh_ritz(H: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD-based Rayleigh-Ritz: returns (sorted_evals, coefficients).

    coefficients shape (n_ritz, n_keep): column i gives the linear combination
    of the input columns of X that produces Ritz vector i.
    """
    S = X.T @ X
    Hproj = X.T @ (H @ X)
    S = 0.5 * (S + S.T)
    Hproj = 0.5 * (Hproj + Hproj.T)

    s_evals, s_evecs = np.linalg.eigh(S)
    # Keep all components with significant overlap.
    threshold = 1e-10 * s_evals.max()
    mask = s_evals > threshold
    n_keep = int(mask.sum())

    U = s_evecs[:, -n_keep:]
    s_keep = s_evals[-n_keep:]
    H_red = U.T @ Hproj @ U
    D_inv_sqrt = np.diag(1.0 / np.sqrt(s_keep))
    A_red = D_inv_sqrt @ H_red @ D_inv_sqrt
    A_red = 0.5 * (A_red + A_red.T)
    evals, evecs = np.linalg.eigh(A_red)
    order = np.argsort(evals)
    coeffs = U @ D_inv_sqrt @ evecs[:, order]
    return evals[order], coeffs


def feast_filter_with_iteration(
    H: np.ndarray,
    window: WindowSpec,
    n_quad: int,
    gamma: float,
    n_ritz: int,
    seed: int,
    feast_iter: int = 1,
    solve_noise: float = 0.0,
) -> np.ndarray:
    """Run FEAST with iteration. Returns final Ritz eigenvalues."""
    z_nodes, w_weights = feast_ellipse_quadrature(window, n_quad, gamma)
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    I = np.eye(n)

    # Initial random starting vectors.
    X = rng.normal(size=(n, n_ritz))
    X /= np.linalg.norm(X, axis=0, keepdims=True)

    for it in range(feast_iter):
        # Contour filter.
        Xf = np.zeros_like(X, dtype=np.complex128)
        for j in range(n_quad):
            z = z_nodes[j]
            w = w_weights[j]
            Y = np.linalg.solve(z * I - H, X)
            if solve_noise > 0:
                Y += solve_noise * rng.normal(size=Y.shape) * np.linalg.norm(Y, axis=0, keepdims=True)
            Xf += 2.0 * np.real(w * Y)

        Xf_real = Xf.real

        # Rayleigh-Ritz extraction.
        evals, coeffs = svd_rayleigh_ritz(H, Xf_real)

        # Build Ritz vectors for next iteration.
        if it < feast_iter - 1:
            n_keep = coeffs.shape[1]
            ritz_vecs = Xf_real @ coeffs  # (n, n_keep)
            ritz_vecs /= np.linalg.norm(ritz_vecs, axis=0, keepdims=True)

            # Pad back to n_ritz with random vectors if needed.
            if n_keep < n_ritz:
                pad = rng.normal(size=(n, n_ritz - n_keep))
                pad /= np.linalg.norm(pad, axis=0, keepdims=True)
                X = np.hstack([ritz_vecs, pad])
            else:
                X = ritz_vecs[:, :n_ritz]

        print(f"  iter {it+1}: Ritz = {np.array2string(evals, precision=6)}")

    return evals


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense FEAST sanity check")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ritz", type=int, default=4)
    parser.add_argument("--n-quad", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--e-min", type=float, default=1.0)
    parser.add_argument("--e-split", type=float, default=2.0)
    parser.add_argument("--e-max", type=float, default=80.0)
    parser.add_argument("--feast-iter", type=int, default=1, help="FEAST subspace iterations")
    parser.add_argument("--solve-noise", type=float, default=0.0,
                        help="Relative noise added to linear solves (simulates GMRES tol)")
    args = parser.parse_args()

    H, eigvals = make_test_matrix(args.n, args.seed, args.e_min, args.e_split, args.e_max)
    w1 = WindowSpec("W1", 0.0, 2.0)

    eig_w1 = eigvals[(eigvals >= w1.a_eV) & (eigvals <= w1.b_eV)]

    print("=== Dense FEAST sanity check ===")
    print(f"Matrix size: {args.n}, Spectrum: [{eigvals.min():.3f}, {eigvals.max():.3f}] eV")
    print(f"Window [{w1.a_eV}, {w1.b_eV}] eV: {eig_w1.size} eigenvalues inside")
    print(f"Exact eigenvalues in window: {np.array2string(eig_w1, precision=6)}")
    print(f"FEAST iterations: {args.feast_iter}, solve_noise: {args.solve_noise}")
    print(f"n_ritz: {args.n_ritz}, n_quad: {args.n_quad}, gamma: {args.gamma}")
    print()

    ritz1 = feast_filter_with_iteration(
        H, w1, args.n_quad, args.gamma, args.n_ritz, args.seed + 1,
        feast_iter=args.feast_iter, solve_noise=args.solve_noise,
    )

    # Error analysis.
    n_cmp = min(len(ritz1), len(eig_w1))
    if n_cmp > 0:
        errors = np.abs(ritz1[:n_cmp] - eig_w1[:n_cmp])
        print(f"\nErrors vs exact (first {n_cmp}): {np.array2string(errors, precision=2, suppress_small=True)}")
        print(f"Max error: {errors.max():.2e} eV")


if __name__ == "__main__":
    main()

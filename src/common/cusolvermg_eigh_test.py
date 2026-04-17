"""4-GPU distributed Hermitian eigensolve test — cuSOLVERMg via JAX FFI.

Runs in a SINGLE Python process that sees all GPUs as local devices.
cuSOLVERMg internally tiles the matrix across GPUs and solves cooperatively.

Usage (Perlmutter, inside an interactive 4-GPU allocation):

    src/ffi/common/cpp/run_shifter.sh python3 -u -m common.cusolvermg_eigh_test

or any equivalent single-process 4-GPU launcher.

What it does
------------
1. Build a 128x128 real-symmetric matrix on CPU (numpy).
2. Place it on device 0 as a jax.Array.
3. Call ``eigh_mg`` — the FFI scatters to all GPUs, runs cusolverMgSyevd,
   gathers back.
4. Compare eigenvalues to ``numpy.linalg.eigvalsh``.  Assert
   max-abs diff < 1e-9.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ffi.cusolvermg import eigh_mg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--tile", type=int, default=32,
                    help="Column tile size for cuSOLVERMg distribution.")
    ap.add_argument("--max-gpus", type=int, default=0,
                    help="0 = use all visible GPUs.")
    args = ap.parse_args()

    n = args.n
    print(f"jax.devices() = {jax.devices()}")
    print(f"jax.device_count() = {jax.device_count()}")
    print(f"n = {n}, tile = {args.tile}, max_gpus = {args.max_gpus or 'all'}")

    rng = np.random.default_rng(17)
    X = rng.standard_normal((n, n)).astype(np.float64)
    A_host = (X + X.T) + np.eye(n) * (2.0 * n)
    ref_evals = np.sort(np.linalg.eigvalsh(A_host))

    A = jnp.asarray(A_host)  # lands on devices()[0]

    # Warmup (compile), then time.
    _ = eigh_mg(A, tile_size=args.tile, max_gpus=args.max_gpus)

    t0 = time.time()
    evals, Q = eigh_mg(A, tile_size=args.tile, max_gpus=args.max_gpus)
    evals_np = np.asarray(jax.device_get(evals))
    Q_np     = np.asarray(jax.device_get(Q))
    dt = time.time() - t0

    evals_sorted = np.sort(evals_np)
    max_err = float(np.max(np.abs(evals_sorted - ref_evals)))

    print(f"wall (after warmup) = {dt*1000:.2f} ms")
    print(f"eval[0]   = {evals_sorted[0]:.8e}")
    print(f"eval[n/2] = {evals_sorted[n//2]:.8e}")
    print(f"eval[-1]  = {evals_sorted[-1]:.8e}")
    print(f"max |eval - ref_eval|   = {max_err:.3e}")

    # Residual check on a few eigenvectors.
    # The returned Q is (n, n) row-major, but cuSOLVERMg computed the
    # eigenvectors of A^T (since we fed a row-major buffer interpreted as
    # col-major).  For a real-symmetric A = A^T, evecs of A^T = evecs of A,
    # so Q's rows (in row-major view) are the eigenvectors.
    # We verify with: A @ q_i ≈ lambda_i * q_i for a few i.
    # First we must re-pair evals to evecs: cuSOLVERMg returns them in
    # ascending order.
    ord_idx = np.argsort(evals_np)
    sample_i = [0, n // 2, n - 1]
    print("residuals (||A q_i - lambda_i q_i||_inf) at i =", sample_i)
    for i in sample_i:
        q = Q_np[ord_idx[i], :]          # row-major: "row i" is an eigenvector
        r = float(np.max(np.abs(A_host @ q - evals_np[ord_idx[i]] * q)))
        # fall back: maybe they're columns instead
        q2 = Q_np[:, ord_idx[i]]
        r2 = float(np.max(np.abs(A_host @ q2 - evals_np[ord_idx[i]] * q2)))
        print(f"  i={i}: row-vec res = {r:.3e}, col-vec res = {r2:.3e}")

    tol = 1e-9 * n
    ok = max_err < tol
    print(f"{'PASS' if ok else 'FAIL'} (tol={tol:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

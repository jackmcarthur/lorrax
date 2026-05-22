"""Correctness test for ``cusolvermp.batched_distributed_solve_lu``.

Mirrors cusolvermp_batched_test.py but targets the general (non-Hermitian)
distributed LU solve used by w_isdf's low_mem path.

Usage::

    lxalloc
    LORRAX_MPI_TYPE=pmix lxrun python3 -u -m common.cusolvermp_solve_lu_test \\
        --nbatch 8 -n 128 --nrhs 128 --mesh 2x2 --dtype c128
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from runtime import set_default_env, init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
set_default_env()
init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cusolvermp import batched_distributed_solve_lu


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_general_batch(nbatch, n, dtype, seed, mesh, identity_only=False):
    """Build (nbatch, n, n) diagonally-dominant random matrix — non-Hermitian
    but well-conditioned so LU is stable.  If ``identity_only``, return
    A = I (sanity-check: X should equal B)."""
    sh = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _b():
        if identity_only:
            A = jnp.broadcast_to(jnp.eye(n, dtype=dtype)[None, :, :],
                                 (nbatch, n, n)).astype(dtype)
            return jax.lax.with_sharding_constraint(A, sh)
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nbatch, n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, n), dtype=jnp.float64)
            A = (a + 1j * b).astype(dtype)
        else:
            A = a.astype(dtype)
        A = A + (2.0 * n) * jnp.eye(n, dtype=dtype)[None, :, :]
        return jax.lax.with_sharding_constraint(A, sh)
    return _b()


def build_rhs_batch(nbatch, n, nrhs, dtype, seed, mesh):
    sh = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _b():
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nbatch, n, nrhs), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, nrhs), dtype=jnp.float64)
            B = (a + 1j * b).astype(dtype)
        else:
            B = a.astype(dtype)
        return jax.lax.with_sharding_constraint(B, sh)
    return _b()


def _parse_mesh(s):
    px, py = s.lower().split("x")
    return int(px), int(py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--nrhs", type=int, default=0, help="rhs cols; default n")
    ap.add_argument("--mesh", type=str, default="2x2")
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identity", action="store_true",
                    help="use A=I — trivial LU; X should equal B exactly")
    args = ap.parse_args()

    Px, Py = _parse_mesh(args.mesh)
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != Px*Py={Px*Py}")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64
    nrhs = args.nrhs or args.n

    if args.n % Px != 0 or args.n % Py != 0 or nrhs % Py != 0:
        _log(f"divisibility fails: N={args.n}%Px={Px}, "
             f"N={args.n}%Py={Py}, NRHS={nrhs}%Py={Py}")
        return 2

    _log(f"=== cusolvermp batched solve_lu: nbatch={args.nbatch} "
         f"n={args.n} nrhs={nrhs} mesh={Px}x{Py} dtype={args.dtype} ===")

    A = build_general_batch(args.nbatch, args.n, dtype, args.seed, mesh,
                             identity_only=args.identity)
    B = build_rhs_batch(args.nbatch, args.n, nrhs, dtype, args.seed + 1, mesh)
    jax.block_until_ready(A); jax.block_until_ready(B)
    multihost_utils.sync_global_devices("inputs_built")

    # Snapshot A before the solve — solve donates A's buffer.
    A_full = multihost_utils.process_allgather(A)
    B_full = multihost_utils.process_allgather(B)

    t0 = time.perf_counter()
    X = batched_distributed_solve_lu(A, B, mesh=mesh)
    jax.block_until_ready(X)
    dt = time.perf_counter() - t0
    _log(f"  solve_lu wall: {dt*1000:.1f} ms")

    X_full = multihost_utils.process_allgather(X)

    if jax.process_index() == 0:
        A_np = np.asarray(A_full)
        B_np = np.asarray(B_full)
        X_np = np.asarray(X_full)

        res_ax_max = 0.0
        for q in range(args.nbatch):
            Aq, Bq, Xq = A_np[q], B_np[q], X_np[q]
            Bnorm = max(np.linalg.norm(Bq), 1.0)
            res_ax = np.linalg.norm(Aq @ Xq - Bq) / Bnorm
            res_ax_max = max(res_ax_max, res_ax)
        _log(f"  max |A X - B|/|B| = {res_ax_max:.3e}")
        # Debug: dump top-left 4x4 of X[0] and B[0] to see the pattern.
        if os.environ.get("LORRAX_LU_DEBUG_DUMP"):
            # Identity sanity: which cells of X match/don't match B?
            # For A=I, expected X = B.  Show mask of zero cells in X.
            diff = X_np[0] - B_np[0]
            _log(f"  |X-B|[0] max = {np.abs(diff).max():.3e}")
            zero_mask = (np.abs(X_np[0]) < 1e-15).astype(int)
            _log(f"  X[0] zero-mask (1 = zero, 0 = nonzero):")
            for row in zero_mask:
                _log("    " + "".join(str(c) for c in row))
        tol = 1e-10
        _log(f"  {'PASS' if res_ax_max < tol else 'FAIL'} at tol {tol:.1e}")
        return 0 if res_ax_max < tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

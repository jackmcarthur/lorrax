"""Correctness test: cuSOLVERMp.batched_distributed_cholesky + batched_distributed_potrs.

4-GPU mesh (default 2x2). Build Nbatch random Hermitian PD matrices each
of size N×N, plus a batched RHS of shape (Nbatch, N, Mrhs), sharded
``P('x', None, 'y')``. Factor via batched_distributed_cholesky, solve
via batched_distributed_potrs. Gather, compare per-slice against
``np.linalg.cholesky`` / ``L_np @ L_np.conj().T @ X - B``.

Usage::

    lxalloc
    lxrun python3 -u -m common.cusolvermp_batched_test \\
        --nbatch 8 -n 128 --mrhs 256 --mesh 2x2 --dtype c128
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
from ffi.cusolvermp import (
    batched_distributed_cholesky,
    batched_distributed_potrs,
)


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_hpd_batch(nbatch, n, dtype, seed, mesh):
    sh = NamedSharding(mesh, P("x", None, "y"))

    @jax.jit
    def _b():
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nbatch, n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, n), dtype=jnp.float64)
            z = (a + 1j * b).astype(dtype)
            H = 0.5 * (z + jnp.conj(jnp.swapaxes(z, -1, -2)))
        else:
            H = 0.5 * (a + jnp.swapaxes(a, -1, -2)).astype(dtype)
        H = H + n * jnp.eye(n, dtype=dtype)[None, :, :]
        return jax.lax.with_sharding_constraint(H, sh)
    return _b()


def build_rhs_batch(nbatch, n, mrhs, dtype, seed, mesh):
    sh = NamedSharding(mesh, P("x", None, "y"))

    @jax.jit
    def _b():
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nbatch, n, mrhs), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, mrhs), dtype=jnp.float64)
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
    ap.add_argument("--mrhs", type=int, default=0, help="rhs cols; default n")
    ap.add_argument("--mesh", type=str, default="2x2")
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--nb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Px, Py = _parse_mesh(args.mesh)
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != Px*Py={Px*Py}")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64
    mrhs = args.mrhs or args.n

    if args.nbatch % Px != 0 or args.n % Py != 0 or mrhs % Py != 0:
        _log(f"divisibility fails: Nbatch={args.nbatch}%Px={Px}, "
             f"N={args.n}%Py={Py}, Mrhs={mrhs}%Py={Py}")
        return 2

    _log(f"=== cusolvermp batched potrf+potrs: nbatch={args.nbatch} "
         f"n={args.n} mrhs={mrhs} mesh={Px}x{Py} dtype={args.dtype} ===")

    A = build_hpd_batch(args.nbatch, args.n, dtype, args.seed, mesh)
    B = build_rhs_batch(args.nbatch, args.n, mrhs, dtype, args.seed + 1, mesh)
    jax.block_until_ready(A); jax.block_until_ready(B)
    multihost_utils.sync_global_devices("inputs_built")

    kw = {"mesh": mesh}
    if args.nb:
        kw["block_size"] = args.nb

    t0 = time.perf_counter()
    L_handle = batched_distributed_cholesky(A, **kw)
    jax.block_until_ready(L_handle.raw)
    dt_potrf = time.perf_counter() - t0
    _log(f"  potrf wall: {dt_potrf*1000:.1f} ms "
         f"(nbatch_local={args.nbatch//Px})")

    t0 = time.perf_counter()
    X = batched_distributed_potrs(L_handle, B, mesh=mesh)
    jax.block_until_ready(X)
    dt_potrs = time.perf_counter() - t0
    _log(f"  potrs wall: {dt_potrs*1000:.1f} ms")

    A_full = multihost_utils.process_allgather(A)
    B_full = multihost_utils.process_allgather(B)
    X_full = multihost_utils.process_allgather(X)

    if jax.process_index() == 0:
        A_np = np.asarray(A_full)
        B_np = np.asarray(B_full)
        X_np = np.asarray(X_full)

        res_ax_max = 0.0
        for q in range(args.nbatch):
            Aq, Bq, Xq = A_np[q], B_np[q], X_np[q]
            Bnorm = max(np.linalg.norm(Bq), 1.0)
            # Verify A X = B directly against the original A (not factored).
            res_ax = np.linalg.norm(Aq @ Xq - Bq) / Bnorm
            res_ax_max = max(res_ax_max, res_ax)
        _log(f"  max |A X - B|/|B| = {res_ax_max:.3e}")
        tol = 1e-10
        _log(f"  {'PASS' if res_ax_max < tol else 'FAIL'} at tol {tol:.1e}")
        return 0 if res_ax_max < tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

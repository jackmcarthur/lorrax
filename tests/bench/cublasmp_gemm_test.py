"""Correctness test for cuBLASMp-backed batched_distributed_gemm.

Usage::
    lxalloc
    LORRAX_MPI_TYPE=pmix lxrun python3 -u tests/bench/cublasmp_gemm_test.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        init_kwargs = {"local_device_ids": [0]} if cvd and "," not in cvd else {}
        try: jax.distributed.initialize(**init_kwargs)
        except Exception: pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cublasmp import batched_distributed_gemm


def _log(s):
    if jax.process_index() == 0: print(s, flush=True)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--nbatch", type=int, default=2)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("--mesh", type=str, default="2x2")
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--transa", default="N")
    ap.add_argument("--transb", default="N")
    args = ap.parse_args()

    Px, Py = map(int, args.mesh.lower().split("x"))
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != {Px*Py}")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64

    m, n, k = args.m, args.n, args.k
    A_rows, A_cols = (m, k) if args.transa == "N" else (k, m)
    B_rows, B_cols = (k, n) if args.transb == "N" else (n, k)

    @jax.jit
    def make():
        kr, ki, br, bi, cr, ci = jax.random.split(jax.random.key(0), 6)
        def cpx(a, b, shape):
            ar = jax.random.normal(a, shape, dtype=jnp.float64)
            ai = jax.random.normal(b, shape, dtype=jnp.float64)
            return (ar + 1j * ai).astype(dtype) if jnp.issubdtype(dtype, jnp.complexfloating) else ar.astype(dtype)
        A = cpx(kr, ki, (args.nbatch, A_rows, A_cols))
        B = cpx(br, bi, (args.nbatch, B_rows, B_cols))
        C = cpx(cr, ci, (args.nbatch, m, n))
        return (jax.lax.with_sharding_constraint(A, sh),
                jax.lax.with_sharding_constraint(B, sh),
                jax.lax.with_sharding_constraint(C, sh))

    A, B, C = make()
    jax.block_until_ready(A); jax.block_until_ready(B); jax.block_until_ready(C)

    A_np = np.asarray(multihost_utils.process_allgather(A))
    B_np = np.asarray(multihost_utils.process_allgather(B))
    C_np = np.asarray(multihost_utils.process_allgather(C))

    alpha = 2.0 + 0.5j if jnp.issubdtype(dtype, jnp.complexfloating) else 2.0
    beta  = 0.3 - 0.1j if jnp.issubdtype(dtype, jnp.complexfloating) else 0.3

    _log(f"=== cublasmp gemm: nbatch={args.nbatch} m={m} n={n} k={k} "
         f"transa={args.transa} transb={args.transb} dtype={args.dtype} ===")

    t0 = time.perf_counter()
    D = batched_distributed_gemm(A, B, C, mesh=mesh,
                                  alpha=alpha, beta=beta,
                                  transa=args.transa, transb=args.transb)
    jax.block_until_ready(D)
    dt = time.perf_counter() - t0
    _log(f"  gemm wall: {dt*1000:.1f} ms")

    D_np = np.asarray(multihost_utils.process_allgather(D))

    if jax.process_index() == 0:
        op = lambda X, t: (X.conj().swapaxes(-1, -2) if t == 'C'
                           else (X.swapaxes(-1, -2) if t == 'T' else X))
        D_ref = alpha * (op(A_np, args.transa) @ op(B_np, args.transb)) + beta * C_np
        err = np.max(np.abs(D_np - D_ref)) / max(np.max(np.abs(D_ref)), 1.0)
        _log(f"  max |D - D_ref|/|D_ref| = {err:.3e}")
        tol = 1e-10 if args.dtype == "c128" else 1e-6
        _log(f"  {'PASS' if err < tol else 'FAIL'} at tol {tol:.1e}")
        return 0 if err < tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

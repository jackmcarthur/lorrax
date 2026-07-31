"""Bench: distributed_cholesky + distributed_trsm timing across mesh shapes.

Single backend (SLATE), one mesh shape per process invocation.  Build a
random Hermitian PD A and an RHS B, both sharded P('x','y') on the
chosen mesh; time potrf and the chained forward-solve trsm separately,
plus a correctness check.

Usage::

    LORRAX_NGPU=4 LORRAX_SELECT_GPU=1 \\
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
        bash src/ffi/common/cpp/run_shifter.sh \\
        python3 -u tests/bench/slate_chol_trsm_bench.py \\
            --mesh 2x2 -n 1024 --dtype c128 --repeats 3
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
    try:
        from ffi.common.ffi_loader import get_lib, platform_from_env
        # Explicit platform: get_lib(None) would initialize the XLA backend
        # (jax.default_backend()) BEFORE jax.distributed.initialize below.
        get_lib(platform_from_env()).lrx_slate_init_mpi()
    except Exception:
        pass
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize(local_device_ids=[0])
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.linalg import backend_module
_slate = backend_module("slate")
distributed_cholesky, distributed_trsm = _slate.distributed_cholesky, _slate.distributed_trsm


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def make_hpd(n, dtype, seed, mesh):
    sharding = NamedSharding(mesh, P("x", "y"))
    @jax.jit
    def _b():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.normal(k_r, (n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (n, n), dtype=jnp.float64)
            z = (a + 1j * b).astype(dtype)
            H = 0.5 * (z + z.conj().T)
        else:
            H = 0.5 * (a + a.T).astype(dtype)
        H = H + n * jnp.eye(n, dtype=dtype)
        return jax.lax.with_sharding_constraint(H, sharding)
    return _b()


def make_rhs(n, m, dtype, seed, mesh):
    sharding = NamedSharding(mesh, P("x", "y"))
    @jax.jit
    def _b():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.normal(k_r, (n, m), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (n, m), dtype=jnp.float64)
            B = (a + 1j * b).astype(dtype)
        else:
            B = a.astype(dtype)
        return jax.lax.with_sharding_constraint(B, sharding)
    return _b()


def _parse_mesh(s: str) -> tuple[int, int]:
    p_str, q_str = s.lower().split("x")
    return int(p_str), int(q_str)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--mesh", type=str, required=True,
                    help="grid as PxQ, e.g. 2x2, 1x4, 4x1")
    ap.add_argument("-n", type=int, default=1024)
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--nb", type=int, default=0,
                    help="SLATE tile size; 0 = n/p (cholesky default)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p, q = _parse_mesh(args.mesh)
    world = jax.process_count()
    if p * q != world:
        _log(f"mesh {p}x{q} != world={world}")
        return 2
    if args.n % p != 0 or args.n % q != 0:
        _log(f"n={args.n} not divisible by {p} and {q}")
        return 2

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64

    _log(f"=== mesh={p}x{q}  n={args.n}  dtype={args.dtype}  "
         f"nb={args.nb or args.n//p}  repeats={args.repeats} ===")
    _log(f"  per-rank A shard: ({args.n//p}, {args.n//q}) "
         f"({args.n//p * args.n//q * (16 if args.dtype=='c128' else 8) / 2**20:.1f} MiB)")

    A = make_hpd(args.n, dtype, args.seed, mesh)
    B = make_rhs(args.n, args.n, dtype, args.seed + 1, mesh)
    jax.block_until_ready(A); jax.block_until_ready(B)
    multihost_utils.sync_global_devices("built")

    chol_kw = {"mesh": mesh}
    if args.nb:
        chol_kw["block_size"] = args.nb

    # Warmup pass (compile + lazy ctx).
    multihost_utils.sync_global_devices("warm_chol_start")
    t0 = time.perf_counter()
    L_handle = distributed_cholesky(A, **chol_kw)
    jax.block_until_ready(L_handle.raw)
    multihost_utils.sync_global_devices("warm_chol_end")
    chol_warm = (time.perf_counter() - t0) * 1000

    multihost_utils.sync_global_devices("warm_trsm_start")
    t0 = time.perf_counter()
    Xw = distributed_trsm(L_handle, B, mesh=mesh, op="N")
    jax.block_until_ready(Xw)
    multihost_utils.sync_global_devices("warm_trsm_end")
    trsm_warm = (time.perf_counter() - t0) * 1000
    _log(f"  warmup:  potrf={chol_warm:.1f} ms  trsm={trsm_warm:.1f} ms")

    chol_t, trsm_t = [], []
    for i in range(args.repeats):
        multihost_utils.sync_global_devices(f"c_{i}_start")
        t0 = time.perf_counter()
        L_handle = distributed_cholesky(A, **chol_kw)
        jax.block_until_ready(L_handle.raw)
        multihost_utils.sync_global_devices(f"c_{i}_end")
        chol_t.append((time.perf_counter() - t0) * 1000)

        multihost_utils.sync_global_devices(f"t_{i}_start")
        t0 = time.perf_counter()
        X = distributed_trsm(L_handle, B, mesh=mesh, op="N")
        jax.block_until_ready(X)
        multihost_utils.sync_global_devices(f"t_{i}_end")
        trsm_t.append((time.perf_counter() - t0) * 1000)

    chol_arr = np.asarray(chol_t)
    trsm_arr = np.asarray(trsm_t)
    _log(f"  potrf : mean={chol_arr.mean():6.1f} ms  "
         f"min={chol_arr.min():6.1f}  max={chol_arr.max():6.1f}")
    _log(f"  trsm  : mean={trsm_arr.mean():6.1f} ms  "
         f"min={trsm_arr.min():6.1f}  max={trsm_arr.max():6.1f}")
    _log(f"  TOTAL : mean={(chol_arr+trsm_arr).mean():6.1f} ms")

    # Correctness on the LAST iteration.
    L_can = L_handle.to_jax_lower()
    A_full = multihost_utils.process_allgather(A)
    B_full = multihost_utils.process_allgather(B)
    L_full = multihost_utils.process_allgather(L_can)
    X_full = multihost_utils.process_allgather(X)
    if jax.process_index() == 0:
        A_np = np.asarray(A_full); B_np = np.asarray(B_full)
        L_np = np.asarray(L_full); X_np = np.asarray(X_full)
        rA = float(np.linalg.norm(L_np @ L_np.conj().T - A_np)
                   / max(np.linalg.norm(A_np), 1.0))
        rB = float(np.linalg.norm(L_np @ X_np - B_np)
                   / max(np.linalg.norm(B_np), 1.0))
        _log(f"  correctness: |L*L^H - A|/|A|={rA:.2e}  "
             f"|L @ X - B|/|B|={rB:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

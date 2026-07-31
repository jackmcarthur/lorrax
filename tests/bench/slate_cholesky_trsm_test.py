"""Correctness test: slate.distributed_cholesky + distributed_trsm.

4-GPU 2x2 mesh.  Build a random Hermitian positive-definite A, factor
A = L L^H via distributed_cholesky.  Gather L, compare against
np.linalg.cholesky.  Then solve L X = B for a sharded B via
distributed_trsm(side='L', uplo='L', op='N'); gather X, compare L @ X
with B.

Usage::

    SLURM_JOBID=... LORRAX_NGPU=4 LORRAX_SELECT_GPU=1 \\
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
        bash src/ffi/common/cpp/run_shifter.sh \\
        python3 -u tests/bench/slate_cholesky_trsm_test.py -n 256 --dtype c128
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
    except Exception as _e:
        print(f"slate_init_mpi skipped: {_e}", flush=True)
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize(local_device_ids=[0])
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.slate import distributed_cholesky, distributed_trsm


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_hpd(n, dtype, seed, mesh):
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
        # Shift diagonal to guarantee positive-definite.
        H = H + n * jnp.eye(n, dtype=dtype)
        return jax.lax.with_sharding_constraint(H, sharding)
    return _b()


def build_rhs(n, m, dtype, seed, mesh):
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


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("-m", type=int, default=0, help="rhs cols; default n")
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--nb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = jax.process_count()
    if world != 4:
        _log(f"world={world} not 4; expect 4 procs for 2x2 mesh")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2),
                axis_names=("x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64
    m = args.m or args.n

    _log(f"=== cholesky+trsm test: n={args.n} m={m} dtype={args.dtype} ===")

    # ---------- Cholesky ----------
    A = build_hpd(args.n, dtype, args.seed, mesh)
    jax.block_until_ready(A)
    multihost_utils.sync_global_devices("A_built")

    kw = {"mesh": mesh}
    if args.nb:
        kw["block_size"] = args.nb

    t0 = time.perf_counter()
    L_handle = distributed_cholesky(A, **kw)
    jax.block_until_ready(L_handle.raw)
    dt = time.perf_counter() - t0
    _log(f"  potrf wall: {dt*1000:.1f} ms (returned {type(L_handle).__name__})")

    # Gather and compare.  Use the canonical L (handle.to_jax_lower()).
    L_canonical_jax = L_handle.to_jax_lower()
    A_full = multihost_utils.process_allgather(A)
    L_raw_full = multihost_utils.process_allgather(L_handle.raw)
    L_full = multihost_utils.process_allgather(L_canonical_jax)
    if jax.process_index() == 0:
        A_np = np.asarray(A_full)
        L_np = np.asarray(L_full)               # canonical from handle
        L_raw_np = np.asarray(L_raw_full)       # raw FFI buffer
        L_ref = np.linalg.cholesky(A_np)
        H_norm = max(np.linalg.norm(A_np), 1.0)
        res_canon = float(np.linalg.norm(L_np @ L_np.conj().T - A_np) / H_norm)
        ref_diff_canon = float(np.linalg.norm(L_np - L_ref) /
                               max(np.linalg.norm(L_ref), 1.0))
        print(f"  CANONICAL (via handle.to_jax_lower()): "
              f"|L*L^H - A|/|A|={res_canon:.3e}  "
              f"|L - L_numpy|/|L_numpy|={ref_diff_canon:.3e}", flush=True)
        # Sanity: is the strict-upper of the raw buffer zero?
        # In SLATE GridOrder::Col with default nb=n/p, rank 2 owns SLATE
        # tile (0, 1) = strict upper.  Its data should now be zero.
        # When gathered, that's JAX block (1, 0) = bottom-left.
        ul = L_raw_np[L_raw_np.shape[0]//2:, :L_raw_np.shape[1]//2]
        print(f"  raw L upper-tile-block (rank 2, bottom-left of gather): "
              f"max|val|={float(np.max(np.abs(ul))):.3e}", flush=True)

    # ---------- trsm: solve L X = B ----------
    B = build_rhs(args.n, m, dtype, args.seed + 1, mesh)
    jax.block_until_ready(B)
    multihost_utils.sync_global_devices("B_built")

    B_full = multihost_utils.process_allgather(B)
    # Forward solve: X_fwd = inv(L) @ B
    t0 = time.perf_counter()
    X_fwd = distributed_trsm(L_handle, B, mesh=mesh, op="N")
    jax.block_until_ready(X_fwd)
    dt_fwd = time.perf_counter() - t0
    X_fwd_full = multihost_utils.process_allgather(X_fwd)
    # Adjoint solve: Y_adj = inv(L^H) @ B
    t0 = time.perf_counter()
    Y_adj = distributed_trsm(L_handle, B, mesh=mesh, op="C")
    jax.block_until_ready(Y_adj)
    dt_adj = time.perf_counter() - t0
    Y_adj_full = multihost_utils.process_allgather(Y_adj)

    if jax.process_index() == 0:
        B_np = np.asarray(B_full); Bnorm = max(np.linalg.norm(B_np), 1.0)
        X_fwd_np = np.asarray(X_fwd_full)
        Y_adj_np = np.asarray(Y_adj_full)
        L_can = L_np  # canonical lower factor (from handle.to_jax_lower())
        res_fwd = float(np.linalg.norm(L_can @ X_fwd_np - B_np) / Bnorm)
        res_adj = float(np.linalg.norm(L_can.conj().T @ Y_adj_np - B_np) / Bnorm)
        _log(f"  trsm forward (op='N'): |L @ X - B|/|B|   = {res_fwd:.3e}  ({dt_fwd*1000:.1f} ms)")
        _log(f"  trsm adjoint (op='C'): |L^H @ Y - B|/|B| = {res_adj:.3e}  ({dt_adj*1000:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

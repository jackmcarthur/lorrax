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
        python3 -u -m common.slate_cholesky_trsm_test -n 256 --dtype c128
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
        from ffi.common.ffi_loader import get_lib
        get_lib().lrx_slate_init_mpi()
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
    ap = argparse.ArgumentParser()
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
    L = distributed_cholesky(A, **kw)
    jax.block_until_ready(L)
    dt = time.perf_counter() - t0
    _log(f"  potrf wall: {dt*1000:.1f} ms")

    # Gather and compare.
    A_full = multihost_utils.process_allgather(A)
    L_full = multihost_utils.process_allgather(L)
    if jax.process_index() == 0:
        A_np = np.asarray(A_full)
        L_np = np.asarray(L_full)
        # Try several transforms to find the layout that matches the
        # numpy reference, same pattern as the eigh test.
        L_ref = np.linalg.cholesky(A_np)
        H_norm = max(np.linalg.norm(A_np), 1.0)
        variants = {
            "L          ": L_np,
            "L.T        ": L_np.T,
            "L.conj().T ": L_np.conj().T,
            "tril(L)    ": np.tril(L_np),
            "tril(L).T  ": np.tril(L_np).T,
            "tril(L.T)  ": np.tril(L_np.T),
            "tril(L.c.T)": np.tril(L_np.conj().T),
        }
        best_name, best_res = None, None
        for name, Lt in variants.items():
            res = float(np.linalg.norm(Lt @ Lt.conj().T - A_np) / H_norm)
            ref_diff = float(np.linalg.norm(Lt - L_ref) / max(np.linalg.norm(L_ref), 1.0))
            print(f"    {name}: |Lt*Lt^H - A|/|A|={res:.3e}  "
                  f"|Lt - L_numpy|/|L_numpy|={ref_diff:.3e}", flush=True)
            if best_res is None or res < best_res:
                best_res, best_name = res, name
        print(f"  best variant: {best_name} (residual {best_res:.3e})",
              flush=True)

    # ---------- trsm: solve L X = B ----------
    B = build_rhs(args.n, m, dtype, args.seed + 1, mesh)
    jax.block_until_ready(B)
    multihost_utils.sync_global_devices("B_built")

    t0 = time.perf_counter()
    X = distributed_trsm(L, B, mesh=mesh, side="L", uplo="L", op="N",
                         block_size=(args.nb or None))
    jax.block_until_ready(X)
    dt = time.perf_counter() - t0
    _log(f"  trsm wall: {dt*1000:.1f} ms")

    B_full = multihost_utils.process_allgather(B)
    X_full = multihost_utils.process_allgather(X)
    if jax.process_index() == 0:
        B_np = np.asarray(B_full)
        X_np = np.asarray(X_full)
        Bnorm = max(np.linalg.norm(B_np), 1.0)
        L_np = np.asarray(L_full)
        # Canonical numpy L is tril(L.T) from the Cholesky probe above.
        L_canonical = np.tril(L_np.T)
        for lhs_name, lhs in [
            ("L_canon @ X       ", L_canonical @ X_np),
            ("L_canon @ X.T     ", L_canonical @ X_np.T),
            ("X @ L_canon       ", X_np @ L_canonical),
            ("X.T @ L_canon     ", X_np.T @ L_canonical),
            ("L @ X             ", L_np @ X_np),
            ("L.T @ X           ", L_np.T @ X_np),
            ("X.T @ L           ", X_np.T @ L_np),
            ("X @ L.T           ", X_np @ L_np.T),
        ]:
            res = float(np.linalg.norm(lhs - B_np) / Bnorm)
            print(f"    |{lhs_name} - B|/|B| = {res:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

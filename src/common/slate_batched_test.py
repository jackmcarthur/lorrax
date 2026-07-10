"""Correctness test: slate.batched_distributed_cholesky + batched_distributed_trsm.

4-GPU mesh (default 2x2).  Build Nbatch random Hermitian PD matrices each
of size N×N, sharded ``P('x', None, 'y')`` so the batch is split across
the X axis and each matrix is distributed across the Y axis.  Factor via
``batched_distributed_cholesky``, then solve ``L_q X_q = B_q`` via
``batched_distributed_trsm``.  Gather, compare per slice against
``np.linalg.cholesky`` / ``L @ X - B``.

Usage::

    LORRAX_NGPU=4 LORRAX_SELECT_GPU=1 LORRAX_MPI_TYPE=cray_shasta \\
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
        bash src/ffi/common/cpp/run_shifter.sh \\
        python3 -u -m common.slate_batched_test \\
        --nbatch 8 -n 128 --mesh 2x2 --dtype c128
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
from ffi.slate import (
    batched_distributed_cholesky,
    batched_distributed_trsm,
)


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_hpd_batch(nbatch, n, dtype, seed, mesh):
    sharding = NamedSharding(mesh, P("x", None, "y"))

    @jax.jit
    def _b():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.normal(k_r, (nbatch, n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, n), dtype=jnp.float64)
            z = (a + 1j * b).astype(dtype)
            H = 0.5 * (z + jnp.conj(jnp.swapaxes(z, -1, -2)))
        else:
            H = 0.5 * (a + jnp.swapaxes(a, -1, -2)).astype(dtype)
        H = H + n * jnp.eye(n, dtype=dtype)[None, :, :]
        return jax.lax.with_sharding_constraint(H, sharding)
    return _b()


def build_rhs_batch(nbatch, n, m, dtype, seed, mesh):
    sharding = NamedSharding(mesh, P("x", None, "y"))

    @jax.jit
    def _b():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.normal(k_r, (nbatch, n, m), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (nbatch, n, m), dtype=jnp.float64)
            B = (a + 1j * b).astype(dtype)
        else:
            B = a.astype(dtype)
        return jax.lax.with_sharding_constraint(B, sharding)
    return _b()


def _parse_mesh(s):
    px, py = s.lower().split("x")
    return int(px), int(py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("-m", type=int, default=0, help="rhs cols; default n")
    ap.add_argument("--mesh", type=str, default="2x2",
                    help="PxxPy, e.g. 2x2, 1x4, 4x1")
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--nb", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Px, Py = _parse_mesh(args.mesh)
    world = jax.process_count()
    if Px * Py != world:
        _log(f"world={world} != Px*Py={Px*Py}; bad mesh")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))

    if args.nbatch % Px != 0:
        _log(f"Nbatch={args.nbatch} must be divisible by Px={Px}")
        return 2
    if args.n % Py != 0:
        _log(f"N={args.n} must be divisible by Py={Py}")
        return 2

    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64
    m = args.m or args.n
    if m % Py != 0:
        _log(f"m={m} must be divisible by Py={Py}")
        return 2

    _log(f"=== batched cholesky+trsm: nbatch={args.nbatch} n={args.n} "
         f"m={m} mesh={Px}x{Py} dtype={args.dtype} ===")

    # ---------- Cholesky ----------
    A = build_hpd_batch(args.nbatch, args.n, dtype, args.seed, mesh)
    jax.block_until_ready(A)
    multihost_utils.sync_global_devices("A_built")

    kw = {"mesh": mesh}
    if args.nb:
        kw["block_size"] = args.nb

    t0 = time.perf_counter()
    L_handle = batched_distributed_cholesky(A, **kw)
    jax.block_until_ready(L_handle.raw)
    dt = time.perf_counter() - t0
    _log(f"  batched potrf wall: {dt*1000:.1f} ms "
         f"(returned {type(L_handle).__name__}, nbatch_local={args.nbatch//Px})")

    A_full = multihost_utils.process_allgather(A)

    # ---------- trsm ----------
    B = build_rhs_batch(args.nbatch, args.n, m, dtype, args.seed + 1, mesh)
    jax.block_until_ready(B)
    multihost_utils.sync_global_devices("B_built")
    B_full = multihost_utils.process_allgather(B)

    t0 = time.perf_counter()
    X_fwd = batched_distributed_trsm(L_handle, B, mesh=mesh, op="N")
    jax.block_until_ready(X_fwd)
    dt_fwd = time.perf_counter() - t0
    X_fwd_full = multihost_utils.process_allgather(X_fwd)

    t0 = time.perf_counter()
    Y_adj = batched_distributed_trsm(L_handle, B, mesh=mesh, op="C")
    jax.block_until_ready(Y_adj)
    dt_adj = time.perf_counter() - t0
    Y_adj_full = multihost_utils.process_allgather(Y_adj)

    if jax.process_index() == 0:
        A_np = np.asarray(A_full)
        B_np = np.asarray(B_full)
        X_fwd_np = np.asarray(X_fwd_full)
        Y_adj_np = np.asarray(Y_adj_full)

        # Per-slice residuals.  L is implicit (handle.raw is layout-mangled
        # bytes); verify via L X = B style checks using per-slice numpy
        # cholesky for cross-check on the fly.
        res_chol_max = 0.0
        res_fwd_max = 0.0
        res_adj_max = 0.0
        for q in range(args.nbatch):
            Aq = A_np[q]
            Bq = B_np[q]
            Xq = X_fwd_np[q]
            Yq = Y_adj_np[q]
            Anorm = max(np.linalg.norm(Aq), 1.0)
            Bnorm = max(np.linalg.norm(Bq), 1.0)
            # Reference numpy Cholesky.
            Lq_ref = np.linalg.cholesky(Aq)
            # Forward: L X should equal B (L from the factorization).  We
            # don't have a gathered canonical L, so compare against numpy
            # Lq_ref — correctness of the solve wrt the true factor.
            res_fwd = np.linalg.norm(Lq_ref @ Xq - Bq) / Bnorm
            res_adj = np.linalg.norm(Lq_ref.conj().T @ Yq - Bq) / Bnorm
            # Cholesky residual using the numpy reference: just a sanity
            # check that Aq is PD (should be ~0).
            res_chol = np.linalg.norm(Lq_ref @ Lq_ref.conj().T - Aq) / Anorm
            res_chol_max = max(res_chol_max, res_chol)
            res_fwd_max  = max(res_fwd_max,  res_fwd)
            res_adj_max  = max(res_adj_max,  res_adj)
        _log(f"  max |L_np * L_np^H - A|/|A|   (PD sanity) = {res_chol_max:.3e}")
        _log(f"  max |L_np * X_fwd - B|/|B|    (op='N')    = {res_fwd_max:.3e}  "
             f"({dt_fwd*1000:.1f} ms)")
        _log(f"  max |L_np^H * Y_adj - B|/|B|  (op='C')    = {res_adj_max:.3e}  "
             f"({dt_adj*1000:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

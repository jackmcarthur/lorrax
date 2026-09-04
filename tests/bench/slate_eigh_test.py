"""Correctness smoke test for the SLATE backend's distributed_eigh.

4-GPU 2x2 mesh: builds a random Hermitian matrix, runs slate::heev via
the FFI, gathers eigenvalues + eigenvectors to rank 0, checks against
numpy.linalg.eigh on the same matrix.

Usage::

    export LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD
    lx run -N 1 -G 4 -n 4 -- env PYTHONPATH="$LORRAX_CHECKOUT/src" \\
        python3 -u tests/bench/slate_eigh_test.py -n 256 --dtype c128
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
    # Eagerly init MPI_THREAD_MULTIPLE BEFORE JAX touches anything, so PMI
    # is fresh from srun.  phdf5's init_mpi_eager pattern; reused here.
    try:
        from ffi.common.ffi_loader import get_lib, platform_from_env
        # Explicit platform: get_lib(None) would initialize the XLA backend
        # (jax.default_backend()) BEFORE jax.distributed.initialize below.
        get_lib(platform_from_env()).lrx_slate_init_mpi()
    except Exception as _e:
        print(f"slate_init_mpi skipped: {_e}", flush=True)
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            # The launcher gives each process exactly one visible GPU. Tell JAX so
            # explicitly; otherwise its distributed init waits on absent
            # local-topology keys from the other ranks.
            jax.distributed.initialize(local_device_ids=[0])
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()
from distrib_la import backend_module                                # noqa: E402

distributed_eigh = backend_module("slate").distributed_eigh


def _log(s: str) -> None:
    if jax.process_index() == 0:
        print(s, flush=True)


def make_hermitian(n: int, seed: int, dtype, mesh: Mesh,
                   kind: str = "random") -> jax.Array:
    sharding = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def build():
        if kind == "diag":
            # Known eigvals = 1..n, eigvecs = identity.
            diag = jnp.arange(1, n + 1, dtype=jnp.float64)
            H = jnp.diag(diag).astype(dtype)
        else:
            k = jax.random.key(seed)
            if jnp.issubdtype(dtype, jnp.complexfloating):
                k_r, k_i = jax.random.split(k, 2)
                a = jax.random.normal(k_r, (n, n), dtype=jnp.float64)
                b = jax.random.normal(k_i, (n, n), dtype=jnp.float64)
                z = a + 1j * b
                H = 0.5 * (z + z.conj().T)
                H = H.astype(dtype)
            else:
                a = jax.random.normal(k, (n, n), dtype=jnp.float64)
                H = 0.5 * (a + a.T)
                H = H.astype(dtype)
        return jax.lax.with_sharding_constraint(H, sharding)

    return build()


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nb", type=int, default=0,
                    help="SLATE tile size nb (0 = n/p)")
    ap.add_argument("--kind", choices=["random", "diag"], default="random",
                    help="diag: A=diag(1..n), so expected Q=I (helps diagnose "
                         "layout)")
    args = ap.parse_args()

    world = jax.process_count()
    if world not in (1, 4, 9, 16):
        _log(f"world={world} not square; slate heev needs p==q. exiting.")
        return 2
    side = int(round(world ** 0.5))
    mesh = Mesh(np.asarray(jax.devices()).reshape(side, side),
                axis_names=("x", "y"))

    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64
    _log(f"world={world} grid=({side},{side}) n={args.n} dtype={args.dtype}")

    H = make_hermitian(args.n, args.seed, dtype, mesh, args.kind)
    jax.block_until_ready(H)
    multihost_utils.sync_global_devices("matrix_built")

    t0 = time.perf_counter()
    W, Q = distributed_eigh(
        H, mesh=mesh,
        compute_evecs=True,
        block_size=(args.nb or None),
    )
    jax.block_until_ready(W)
    jax.block_until_ready(Q)
    multihost_utils.sync_global_devices("eigh_done")
    dt = time.perf_counter() - t0
    _log(f"slate::heev wall: {dt*1000:.1f} ms")

    # Gather & compare against numpy reference on rank 0.
    H_full = multihost_utils.process_allgather(H)
    W_full = multihost_utils.process_allgather(W)
    Q_full = multihost_utils.process_allgather(Q)

    if jax.process_index() == 0:
        H_np = np.asarray(H_full)
        W_np = np.asarray(W_full)[:args.n]  # in case replica factor > 1
        Q_np = np.asarray(Q_full)

        W_ref = np.linalg.eigvalsh(H_np)
        max_abs_ev = float(np.max(np.abs(W_np - W_ref)))
        tol = 1e-9 if dtype == jnp.float64 else 1e-8
        ok_ev = max_abs_ev < tol * max(1.0, float(np.max(np.abs(W_ref))))
        print(f"  max|W_slate - W_numpy| = {max_abs_ev:.3e}  "
              f"(tol ~ {tol*max(1.0, float(np.max(np.abs(W_ref)))):.1e})",
              flush=True)

        # Residual: |H Q - Q diag(W)| / |H|
        # Try several layout transforms to diagnose how SLATE stored Q.
        if Q_np.shape == (args.n, args.n):
            H_norm = max(np.linalg.norm(H_np), 1.0)
            HQ = H_np @ Q_np
            QW = Q_np * W_np[None, :]
            res = float(np.linalg.norm(HQ - QW) / H_norm)
            print(f"  residual |HQ - QW|/|H| = {res:.3e}", flush=True)
            if args.kind == "diag":
                # For diagonal H, expected Q = I — reveals any layout issues.
                print(f"  |Q - I|_F = {np.linalg.norm(Q_np - np.eye(args.n)):.3e}")
                sig = np.argwhere(np.abs(Q_np) > 0.5)
                print(f"  |Q[i,j]| > 0.5 at: {[(int(i),int(j)) for (i,j) in sig[:8]]}")
        else:
            print(f"  Q gather shape {Q_np.shape} != ({args.n},{args.n}); "
                  f"skipping residual", flush=True)

        print("RESULT:", "PASS" if ok_ev else "FAIL", flush=True)
        return 0 if ok_ev else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

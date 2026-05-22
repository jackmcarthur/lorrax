"""SLATE heev vs cuSOLVERMp syevd timing — 4 GPUs / 2x2 grid, complex128.

Runs ONE backend per invocation (each needs different process affinity):

    # cuSOLVERMp (default, all GPUs visible per process)
    LORRAX_NGPU=4 bash src/ffi/common/cpp/run_shifter.sh \\
        python3 -u -m common.slate_vs_cusolvermp_bench \\
        --backend cusolvermp -n 2048 --repeats 5

    # SLATE (needs 1-GPU-per-process via LORRAX_SELECT_GPU=1)
    LORRAX_NGPU=4 LORRAX_SELECT_GPU=1 bash src/ffi/common/cpp/run_shifter.sh \\
        python3 -u -m common.slate_vs_cusolvermp_bench \\
        --backend slate -n 2048 --repeats 5

Warms up once, then reports per-call wall-clock for `repeats` iterations.
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


def _log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


# NOTE: keeps inline dist-init because init order is backend-dependent (slate needs local_device_ids=[0]; cusolvermp uses default all-visible-GPUs path)
def _init_distributed(backend: str) -> None:
    """JAX distributed init — per-backend CUDA affinity assumptions."""
    if os.environ.get("_LORRAX_JAX_DISTRIBUTED_DONE"):
        return
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    if world > 1:
        if backend == "slate":
            # Each process has CUDA_VISIBLE_DEVICES=$SLURM_LOCALID via
            # run_shifter.sh select_gpu.sh — tell JAX it owns exactly 1
            # local GPU (at index 0).
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()
    os.environ["_LORRAX_JAX_DISTRIBUTED_DONE"] = "1"


def build_hermitian(n: int, dtype, seed: int, mesh) -> jax.Array:
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharding = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def _build():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.normal(k_r, (n, n), dtype=jnp.float64)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            b = jax.random.normal(k_i, (n, n), dtype=jnp.float64)
            z = (a + 1j * b).astype(dtype)
            H = 0.5 * (z + z.conj().T)
        else:
            H = 0.5 * (a + a.T).astype(dtype)
        return jax.lax.with_sharding_constraint(H, sharding)

    return _build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cusolvermp", "slate"], required=True)
    ap.add_argument("-n", type=int, default=2048)
    ap.add_argument("--dtype", choices=["f64", "c128"], default="c128")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nb", type=int, default=0,
                    help="SLATE tile size nb; 0 = n/p default")
    args = ap.parse_args()

    _init_distributed(args.backend)
    from jax.sharding import Mesh
    from jax.experimental import multihost_utils

    world = jax.process_count()
    if world != 4:
        _log(f"ERROR: expected 4 procs, got {world}")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2),
                axis_names=("x", "y"))
    dtype = jnp.complex128 if args.dtype == "c128" else jnp.float64

    _log(f"backend={args.backend}  n={args.n}  dtype={args.dtype}  "
         f"repeats={args.repeats}")

    # Build A once, shared across warmup + timed iters.
    A = build_hermitian(args.n, dtype, args.seed, mesh)
    jax.block_until_ready(A)
    multihost_utils.sync_global_devices("built")

    # Pick the eigh call.
    if args.backend == "slate":
        from ffi.slate import distributed_eigh
        kw = {"mesh": mesh, "compute_evecs": True}
        if args.nb:
            kw["block_size"] = args.nb
        def eigh_call():
            return distributed_eigh(A, **kw)
    else:
        from ffi.cusolvermp import distributed_eigh
        kw = {"mesh": mesh, "compute_evecs": True}
        if args.nb:
            kw["block_size"] = args.nb
        def eigh_call():
            return distributed_eigh(A, **kw)

    # Warmup (triggers compile + lazy context init).
    t0 = time.perf_counter()
    out = eigh_call()
    jax.block_until_ready(out)
    multihost_utils.sync_global_devices("warm")
    warm_ms = (time.perf_counter() - t0) * 1000
    _log(f"  warmup: {warm_ms:.1f} ms")

    # Timed iterations.
    samples = []
    for i in range(args.repeats):
        multihost_utils.sync_global_devices(f"start_{i}")
        t0 = time.perf_counter()
        out = eigh_call()
        jax.block_until_ready(out)
        multihost_utils.sync_global_devices(f"end_{i}")
        samples.append((time.perf_counter() - t0) * 1000)
        _log(f"  iter {i}: {samples[-1]:.1f} ms")

    arr = np.asarray(samples)
    _log(f"SUMMARY backend={args.backend} n={args.n} dtype={args.dtype}: "
         f"mean={arr.mean():.1f} ms  min={arr.min():.1f} ms  "
         f"max={arr.max():.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

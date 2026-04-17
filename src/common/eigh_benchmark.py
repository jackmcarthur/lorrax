"""Eigh timing benchmark — 1 GPU vs 4 GPUs at n=2048.

Run in two phases:

  # phase A: single-process, 1 / 4 GPU via cusolverMg + jnp.linalg.eigh baseline
  LORRAX_NGPU=4 LORRAX_NTASKS=1 \\
      src/ffi/common/cpp/run_shifter.sh \\
      python3 -u -m common.eigh_benchmark --mode single -n 2048 --repeats 5

  # phase B: multi-process, cusolverMp on a 2x2 grid
  LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
      CUSOLVERMP_FORCE_NCCL=1 \\
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \\
      XLA_PYTHON_CLIENT_PREALLOCATE=false \\
      python3 -u -m common.eigh_benchmark --mode mp -n 2048 --repeats 5

Prints per-call wall time (ms) with one warm-up excluded so JIT compile and
lazy context setup don't show up in the mean.
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


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def make_symmetric(n: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n))
    return (X + X.T + np.eye(n) * (2.0 * n)).astype(np.float64)


def time_call(fn, args, repeats: int, name: str) -> None:
    """Warmup + timed repeats.  `fn(*args)` must return a jax.Array (or tuple)."""
    # warmup (triggers compile + lazy init)
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    warmup_ms = (time.perf_counter() - t0) * 1000
    _log(f"  [{name}] warmup = {warmup_ms:.1f} ms")

    samples = []
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000)
    samples_arr = np.asarray(samples)
    _log(
        f"  [{name}] steady-state (n={len(samples)}): "
        f"mean = {samples_arr.mean():.1f} ms, "
        f"std = {samples_arr.std():.1f} ms, "
        f"min = {samples_arr.min():.1f} ms, "
        f"max = {samples_arr.max():.1f} ms"
    )


def run_single_process(n: int, repeats: int) -> int:
    """Benchmark jnp.linalg.eigh (1 GPU) and eigh_mg (multi-GPU)."""
    from ffi.cusolvermg import eigh_mg

    _log(f"\n=== single-process mode, n={n}, repeats={repeats} ===")
    _log(f"jax.devices() = {jax.devices()}")

    A_host = make_symmetric(n)
    # Correctness reference (cheap at n=2048)
    ref = np.sort(np.linalg.eigvalsh(A_host))

    # 1) jnp.linalg.eigh on 1 GPU (device 0 only)
    A1 = jnp.asarray(A_host)   # lands on devices()[0]
    jit_eigh = jax.jit(jnp.linalg.eigh)
    _log("\n--- jnp.linalg.eigh  (1 GPU, device 0) ---")
    time_call(lambda a: jit_eigh(a), (A1,), repeats, "jnp.eigh-1GPU")
    evals1, _ = jit_eigh(A1)
    evals1_np = np.sort(np.asarray(jax.device_get(evals1)))
    _log(f"  max |eigvals - ref| = {np.max(np.abs(evals1_np - ref)):.2e}")

    # 2) cusolverMg with max_gpus=1 (single-device path through the FFI)
    _log("\n--- cusolverMg eigh_mg (max_gpus=1) ---")
    tile = 256
    time_call(lambda a: eigh_mg(a, tile_size=tile, max_gpus=1),
              (A1,), repeats, "Mg-1GPU")
    evals_mg1, _ = eigh_mg(A1, tile_size=tile, max_gpus=1)
    evals_mg1_np = np.sort(np.asarray(jax.device_get(evals_mg1)))
    _log(f"  max |eigvals - ref| = {np.max(np.abs(evals_mg1_np - ref)):.2e}")

    # 3) cusolverMg with all GPUs
    _log(f"\n--- cusolverMg eigh_mg (max_gpus=0, all={jax.device_count()}) ---")
    time_call(lambda a: eigh_mg(a, tile_size=tile, max_gpus=0),
              (A1,), repeats, f"Mg-{jax.device_count()}GPU")
    evals_mgN, _ = eigh_mg(A1, tile_size=tile, max_gpus=0)
    evals_mgN_np = np.sort(np.asarray(jax.device_get(evals_mgN)))
    _log(f"  max |eigvals - ref| = {np.max(np.abs(evals_mgN_np - ref)):.2e}")

    return 0


def run_multiprocess(n: int, repeats: int) -> int:
    """Benchmark cusolverMp distributed_eigh on a 2x2 grid."""
    # multi-process JAX bootstrap
    _DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
    if not os.environ.get(_DIST_SENTINEL):
        if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
            try:
                jax.distributed.initialize()
            except Exception:
                pass
        os.environ[_DIST_SENTINEL] = "1"

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi.cusolvermp import distributed_eigh

    world = jax.process_count()
    _log(f"\n=== multi-process mode, n={n}, repeats={repeats}, world={world} ===")

    p = int(np.sqrt(world))
    q = world // p
    if p * q != world:
        _log(f"ERROR: world={world} not a square; pick 1/4/9/16 procs")
        return 2
    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, ("x", "y"))

    A_host = make_symmetric(n)
    ref = np.sort(np.linalg.eigvalsh(A_host))

    A = jax.device_put(jnp.asarray(A_host),
                       NamedSharding(mesh, P("x", "y")))

    _log(f"\n--- cusolverMp distributed_eigh ({p}x{q} grid) ---")
    time_call(lambda a: distributed_eigh(a, mesh=mesh),
              (A,), repeats, f"Mp-{world}proc")

    # correctness
    evals, _ = distributed_eigh(A, mesh=mesh)
    from jax.experimental import multihost_utils
    evals_np = np.sort(np.asarray(multihost_utils.process_allgather(evals, tiled=False)))
    _log(f"  max |eigvals - ref| = {np.max(np.abs(evals_np - ref)):.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "mp"], required=True)
    ap.add_argument("-n", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "single":
        return run_single_process(args.n, args.repeats)
    else:
        return run_multiprocess(args.n, args.repeats)


if __name__ == "__main__":
    sys.exit(main())

"""Block-size sweep for cuSOLVERMp distributed_eigh.

For each (n, block_size), times steady-state wall across `repeats` calls
after one warmup.  Block sizes < n/p produce a correctness-preserving
(for eigenvalues) block-cyclic layout mismatch — the eigenvectors will
be permuted relative to the input basis, but eigenvalues still match
numpy to ~1e-10.
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


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True,
                    help="Block sizes to sweep.")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi.cusolvermp import distributed_eigh

    world = jax.process_count()
    p = int(np.sqrt(world))
    q = world // p
    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, ("x", "y"))

    rng = np.random.default_rng(17)
    X = rng.standard_normal((args.n, args.n))
    A_host = (X + X.T + np.eye(args.n) * (2.0 * args.n)).astype(np.float64)
    ref = np.sort(np.linalg.eigvalsh(A_host))
    A = jax.device_put(jnp.asarray(A_host), NamedSharding(mesh, P("x", "y")))

    _log(f"n={args.n}  grid={p}x{q}  repeats={args.repeats}")
    _log(f"{'block':>6}  {'warmup':>9}  {'mean':>8}  {'std':>6}  {'min':>8}  "
         f"{'max|evals-ref|':>14}")
    for block in args.blocks:
        if args.n % (block * p) != 0 or args.n % (block * q) != 0:
            _log(f"  block={block:<4}   skip: n%block*{p}!=0")
            continue

        try:
            # warmup
            t0 = time.perf_counter()
            out = distributed_eigh(A, mesh=mesh, block_size=block)
            jax.block_until_ready(out)
            warmup_ms = (time.perf_counter() - t0) * 1000

            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = distributed_eigh(A, mesh=mesh, block_size=block)
                jax.block_until_ready(out)
                samples.append((time.perf_counter() - t0) * 1000)

            evals, _ = distributed_eigh(A, mesh=mesh, block_size=block)
            from jax.experimental import multihost_utils
            evals_np = np.sort(np.asarray(
                multihost_utils.process_allgather(evals, tiled=False)))
            err = float(np.max(np.abs(evals_np - ref)))

            s = np.asarray(samples)
            _log(f"{block:>6}  {warmup_ms:>8.0f}  {s.mean():>7.0f}  "
                 f"{s.std():>5.0f}  {s.min():>7.0f}  {err:>14.2e}")
        except Exception as e:
            _log(f"  block={block:<4}   ERROR: {type(e).__name__}: "
                 f"{str(e)[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

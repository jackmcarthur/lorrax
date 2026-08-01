"""Block-size sweep for cuSOLVERMp distributed_eigh.

For each (n, block_size), times steady-state wall across `repeats` calls
after one warmup.  Block sizes < n/p produce a correctness-preserving
(for eigenvalues) block-cyclic layout mismatch — the eigenvectors will
be permuted relative to the input basis, but eigenvalues still match
numpy to ~1e-10.
"""
from __future__ import annotations

import argparse
import sys
import time

# Env defaults + jax.distributed + CPU fallback, BEFORE `import jax`.
from runtime import bootstrap                                 # noqa: E402
bootstrap()

import numpy as np                                            # noqa: E402
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from common.collectives import (all_gather_processes,          # noqa: E402
                                device_put_process_local,
                                process_rank, resolve_mesh)


def _log(msg: str) -> None:
    if process_rank() == 0:
        print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True,
                    help="Block sizes to sweep.")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi.linalg import backend_module
    distributed_eigh = backend_module("cusolvermp").distributed_eigh

    # ``resolve_mesh`` builds the square s x s mesh (square-only ruling,
    # repo docs/architecture/decisions.md 2026-08-01; a non-square device
    # count refuses, naming the square count to request) and refuses a mesh
    # this process owns no device in.
    mesh = resolve_mesh()
    p, q = (int(s) for s in mesh.devices.shape)

    rng = np.random.default_rng(17)
    X = rng.standard_normal((args.n, args.n))
    A_host = (X + X.T + np.eye(args.n) * (2.0 * args.n)).astype(np.float64)
    ref = np.sort(np.linalg.eigvalsh(A_host))
    # Process-local (AA.1): plain device_put of the host operand would pay
    # a P × n² × 8 B assert_equal all-gather and distort the benchmark.
    A = device_put_process_local(A_host, NamedSharding(mesh, P("x", "y")))

    _log(f"n={args.n}  grid={p}x{q}  repeats={args.repeats}")
    _log(f"{'block':>6}  {'warmup':>9}  {'mean':>8}  {'std':>6}  {'min':>8}  "
         f"{'max|evals-ref|':>14}")
    n_failed = 0
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
            evals_np = np.sort(all_gather_processes(evals))
            err = float(np.max(np.abs(evals_np - ref)))

            s = np.asarray(samples)
            _log(f"{block:>6}  {warmup_ms:>8.0f}  {s.mean():>7.0f}  "
                 f"{s.std():>5.0f}  {s.min():>7.0f}  {err:>14.2e}")
        except Exception as e:
            # Swallow-and-continue is right for a *sweep* (one block size
            # failing must not hide the others), but the sweep's exit code
            # must still report it: previously every block could error and
            # the process still exited 0, so a CI invocation of this script
            # could never fail.
            n_failed += 1
            _log(f"  block={block:<4}   ERROR: {type(e).__name__}: "
                 f"{str(e)[:120]}")

    if n_failed:
        _log(f"FAILED: {n_failed} of {len(args.blocks)} block sizes errored")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

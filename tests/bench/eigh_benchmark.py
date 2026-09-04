"""Eigh timing benchmark — cusolverMp distributed_eigh at n=2048.

LORRAX is one JAX process per GPU, so the only mode is multi-process
cusolverMp on a 2x2 grid:

  export LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD
  lx run -N 1 -G 4 -n 4 -- env PYTHONPATH="$LORRAX_CHECKOUT/src" \\
      python3 -u tests/bench/eigh_benchmark.py -n 2048 --repeats 5

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
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import plan as linalg_plan

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

    # Process-local (AA.1): plain device_put of the host operand would pay
    # a P × n² × 8 B assert_equal all-gather and distort the benchmark.
    from common.collectives import device_put_process_local
    A = device_put_process_local(A_host,
                                 NamedSharding(mesh, P("x", "y")))

    # Resolved once, outside the timing loop (scorecard L §6).
    eigh_plan = linalg_plan("eigh", mesh, backend="cusolvermp", n=n)
    _log(f"\n--- {eigh_plan.describe()} ({p}x{q} grid) ---")
    time_call(lambda a: eigh_plan(a), (A,), repeats, f"Mp-{world}proc")

    # correctness
    evals, _ = eigh_plan(A)
    from jax.experimental import multihost_utils
    evals_np = np.sort(np.asarray(multihost_utils.process_allgather(evals, tiled=False)))
    _log(f"  max |eigvals - ref| = {np.max(np.abs(evals_np - ref)):.2e}")
    return 0


def run_dispatch(sizes, repeats: int, batch: int, backend: str) -> int:
    """The dispatch question: q-BATCHED native vs ONE distributed tile.

    This is the measurement ``distrib_la.Plan`` cites for
    choosing a backend, and the one ``bandstructure.bse_setup`` faces per q.
    Both arms decompose the SAME kind of matrix — Hermitian complex128,
    ``(n, n)`` — and are reported as **ms per matrix**, the only comparable
    unit:

      native  ``jnp.linalg.eigh`` on ``(batch, n, n)`` sharded
              ``P(('x','y'), None, None)``: each device solves
              ``batch/ndev`` WHOLE matrices concurrently.  Costs
              ``batch/ndev · n²·16`` bytes per device, so it stops fitting
              first.
      ffi     a resolved ``distrib_la`` PLAN on ONE ``(n, n)`` sharded
              ``P('x','y')``: the whole mesh works on one matrix,
              ``n²·16/ndev`` per device.

    The plan is built ONCE, outside the timing loop — resolution probes
    the FFI library and the first call builds a BLACS / cuSOLVERMp
    context and compiles an XLA module (1.4–2.7 s, scorecard L §5), and
    neither belongs in a per-call median.

    One JAX process per GPU (the LORRAX process model), square mesh.
    """
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import plan as linalg_plan

    world = jax.process_count()
    p = int(round(np.sqrt(world)))
    if p * p != world:
        _log(f"ERROR: world={world} is not a square process mesh")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(p, p), ("x", "y"))
    ndev = int(mesh.devices.size)
    _log(f"\n=== dispatch mode: {p}x{p} mesh, {ndev} devices, "
         f"batch={batch}, complex128 ===")
    _log(f"{'n':>7} {'native ms/mat':>15} {'ffi ms/mat':>13} "
         f"{'ffi/native':>11}  (batch/ndev = {batch // ndev} per device)")

    rng = np.random.default_rng(3)
    for n in sizes:
        if n % p:
            _log(f"{n:>7}  skipped (n not divisible by mesh axis {p})")
            continue
        z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        A_np = (0.5 * (z + np.conj(z.T)) + n * np.eye(n)).astype(np.complex128)

        # Process-local (AA.1) — see above; the np.broadcast_to view also
        # keeps each rank's host materialisation to its own shard.
        from common.collectives import device_put_process_local
        A_b = device_put_process_local(
            np.broadcast_to(A_np, (batch, n, n)),
            NamedSharding(mesh, P(("x", "y"), None, None)))
        f_nat = jax.jit(jnp.linalg.eigh)
        t_nat = _median_ms(lambda: f_nat(A_b), repeats) / batch

        A_1 = device_put_process_local(A_np, NamedSharding(mesh, P("x", "y")))
        eigh_plan = linalg_plan("eigh", mesh, backend=backend, n=n)
        t_ffi = _median_ms(lambda: eigh_plan(A_1), repeats)

        _log(f"{n:>7} {t_nat:>15.1f} {t_ffi:>13.1f} {t_ffi / t_nat:>11.2f}")
    return 0


def _median_ms(fn, repeats: int) -> float:
    """Warm-up excluded; median of ``repeats`` wall times in ms."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--mode", choices=["mp", "dispatch"], default="mp")
    ap.add_argument("-n", type=int, default=2048)
    ap.add_argument("--sizes", default="512,1024,2048,4096",
                    help="dispatch mode: comma-separated matrix sizes")
    ap.add_argument("--batch", type=int, default=32,
                    help="dispatch mode: q per native batch "
                         "(bse_setup's batch_size default)")
    ap.add_argument("--backend", default="cusolvermp",
                    choices=("cusolvermp", "slate"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "dispatch":
        from runtime import init_jax_distributed
        init_jax_distributed()
        return run_dispatch([int(v) for v in args.sizes.split(",")],
                            args.repeats, args.batch, args.backend)
    return run_multiprocess(args.n, args.repeats)


if __name__ == "__main__":
    sys.exit(main())

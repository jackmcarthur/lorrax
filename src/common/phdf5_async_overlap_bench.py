"""Measure read/compute overlap of PhdfWfnReader's async FFI path.

Two modes of the same N-band-chunk loop:

  * ``sync``  — block_until_ready after every ``coeffs_gspace`` call,
                so XLA can't schedule the next read against the current
                chunk's compute.  Wall ≈ N · (read + compute).
  * ``async`` — let jax dispatch the next chunk's read right away;
                only block at the end of the loop.  Wall ≈ N · max(read,
                compute) once the pipeline is full.

Compute is a fixed-size jit'd matmul that targets a set wall time
(``--compute-ms``).  Read wall comes from the handler-side timing
(``LORRAX_PHDF5_TIME=1`` prints per-rank totals).

For the flag to prove overlap, the async wall needs to be close to
``max(N·read, N·compute)`` and the sync wall to ``N·(read+compute)``.
With MoS2 3×3 + a 60 ms compute kernel that's a ~20% end-to-end
speedup in async mode — small but measurable.

Usage (4-GPU)::

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        lxrun python3 -u -m common.phdf5_async_overlap_bench \
            --band-chunk-size 20 --n-chunks 5 --compute-ms 60
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.phdf5_wfn_reader import PhdfWfnReader


def log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def sync_all(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def build_compute_kernel(
    mesh: Mesh, shape: tuple[int, int, int, int, int, int], target_ms: float,
):
    """Build a jit'd matmul that chews roughly ``target_ms``
    milliseconds per call, so loops of read+compute expose overlap."""
    # Calibrate n_iters: tune so one call takes ~target_ms.  A few
    # warm runs amortise compile cost; we pick a conservative default
    # and leave tuning to the caller.
    nk, nb, ns, nx, ny, nz = shape
    # ~256×256 matmul per (k, b, s, x) row, repeated ``n_reps`` times, is
    # a cheap-ish synthetic GPU workload whose time scales ~linearly with
    # n_reps.  We calibrate below to hit roughly ``target_ms``.
    inner = 256

    def make_kernel(n_reps: int):
        def kernel(psi_G):
            # Project onto an (inner,) reduction; matmul that chain.
            x = jnp.reshape(psi_G, (nk * nb * ns * nx * ny, nz)).astype(jnp.complex128)
            W = jnp.eye(nz, inner, dtype=jnp.complex128)
            y = x @ W  # (…, inner)
            V = jnp.eye(inner, inner, dtype=jnp.complex128)
            for _ in range(n_reps):
                y = y @ V
            # Project back so output shape matches input (XLA can't DCE).
            Wback = jnp.eye(inner, nz, dtype=jnp.complex128)
            x = y @ Wback
            return jnp.reshape(x, (nk, nb, ns, nx, ny, nz))
        return jax.jit(kernel)

    dummy = jax.device_put(
        jnp.zeros(shape, dtype=jnp.complex128),
        NamedSharding(mesh, P(None, ("x", "y"), None, None, None, None)))

    # Calibrate: try a few n_reps, pick one that lands closest to target_ms.
    best_kernel = None
    best_ms = None
    for n_reps in (1, 5, 20, 50, 100, 200, 400):
        k = make_kernel(n_reps)
        jax.block_until_ready(k(dummy))  # warmup compile
        t0 = time.perf_counter()
        jax.block_until_ready(k(dummy))
        ms = (time.perf_counter() - t0) * 1e3
        if best_ms is None or abs(ms - target_ms) < abs(best_ms - target_ms):
            best_ms = ms
            best_kernel = k
        if ms > target_ms * 1.5:
            break
    log(f"  calibrated compute kernel: {best_ms:.1f} ms/call "
        f"(target {target_ms:.0f} ms)")
    return best_kernel


def run_loop(
    reader: PhdfWfnReader,
    compute_kernel,
    band_ranges: list[tuple[int, int]],
    mode: str,
) -> float:
    sync_all(f"{mode}_start")
    t0 = time.perf_counter()
    results = []
    for br in band_ranges:
        psi_G = reader.coeffs_gspace(br)
        result = compute_kernel(psi_G)
        if mode == "sync":
            jax.block_until_ready(result)
        results.append(result)
    if mode == "async":
        # Only block once, at the end.
        jax.block_until_ready(results[-1])
    wall = time.perf_counter() - t0
    sync_all(f"{mode}_end")
    return wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wfn",
        default=("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                 "02_mos2_3x3_nosym/qe/nscf/WFN.h5"))
    ap.add_argument("--band-chunk-size", type=int, default=20,
                    help="bands per coeffs_gspace call (must divide world).")
    ap.add_argument("--n-chunks", type=int, default=5)
    ap.add_argument("--compute-ms", type=float, default=60.0,
                    help="target ms per compute kernel call (for calibration).")
    ap.add_argument("--iters", type=int, default=3,
                    help="benchmark iters per mode.")
    args = ap.parse_args()

    world = jax.process_count()
    p, q = (2, 2) if world == 4 else (world, 1)
    mesh = Mesh(np.asarray(jax.devices()).reshape(p, q),
                axis_names=("x", "y"))

    log(f"world={world}  wfn={args.wfn}")
    log(f"band_chunk_size={args.band_chunk_size}  n_chunks={args.n_chunks}")

    reader = PhdfWfnReader(args.wfn, mesh=mesh)
    try:
        nb_total = args.band_chunk_size * args.n_chunks
        if nb_total > reader.nbands:
            log(f"ERROR: want {nb_total} bands but wfn has only {reader.nbands}")
            return 1
        band_ranges = [
            (i * args.band_chunk_size, (i + 1) * args.band_chunk_size)
            for i in range(args.n_chunks)
        ]

        # Output shape of coeffs_gspace — we need it for the compute
        # kernel's input contract.
        shape = (reader.nk_full, args.band_chunk_size,
                 reader.nspinor, *reader.fft_grid)
        compute = build_compute_kernel(mesh, shape, args.compute_ms)

        # Warmup both modes (one iter each) so jit caches are hot.
        log("\n--- warmup ---")
        for mode in ("sync", "async"):
            wall = run_loop(reader, compute, band_ranges, mode)
            log(f"  {mode:<5}: {wall * 1e3:7.1f} ms")

        log(f"\n--- timed iters ({args.iters}) ---")
        sync_walls, async_walls = [], []
        for it in range(args.iters):
            s = run_loop(reader, compute, band_ranges, "sync")
            a = run_loop(reader, compute, band_ranges, "async")
            sync_walls.append(s)
            async_walls.append(a)
            log(f"  [iter {it}]  sync={s * 1e3:7.1f} ms   "
                f"async={a * 1e3:7.1f} ms   "
                f"speedup={s / a:.2f}x")

        log("\n=== summary ===")
        log(f"  sync  mean: {np.mean(sync_walls) * 1e3:7.1f} ms")
        log(f"  async mean: {np.mean(async_walls) * 1e3:7.1f} ms")
        log(f"  speedup:    {np.mean(sync_walls) / np.mean(async_walls):.2f}x")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

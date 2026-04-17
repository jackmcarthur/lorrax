"""Benchmark: parallel FFI write vs gather-and-rank0 write.

This is the apples-to-apples comparison that matters for the zeta
write pattern in isdf_fitting.  We build a large sharded complex128
matrix, then write it two ways:

  (G) gather via multihost_utils.process_allgather → rank 0 h5py write
      (this is what src/common/isdf_fitting.py currently does at
      lines ~1075-1085)

  (F) our parallel-HDF5 FFI via write_sharded_slab

Times each variant ITERS times, reports mean wall time plus effective
bandwidth (GB/s) of the sharded matrix through the write path.

Usage (inside a 4-node / 16-GPU interactive alloc):

    LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \\
        src/ffi/common/cpp/run_shifter.sh env \\
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_vs_gather_bench -n 16384 --iters 3
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

_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init():
    if os.environ.get(_DIST_SENTINEL):
        return
    n = int(os.environ.get("SLURM_NTASKS", "1"))
    if n > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.phdf5 import open_file, write_sharded_slab, close_file


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_matrix_c128(n: int, seed: int, mesh: Mesh) -> jax.Array:
    """Random complex128 matrix sharded P('x','y'); mimics the per-chunk
    zeta shape in isdf_fitting (C128, sharded across the mesh)."""
    sharding = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def _build():
        k = jax.random.key(seed)
        k_r, k_i = jax.random.split(k, 2)
        a = jax.random.uniform(k_r, (n, n), dtype=jnp.float64)
        b = jax.random.uniform(k_i, (n, n), dtype=jnp.float64)
        z = a + 1j * b
        return jax.lax.with_sharding_constraint(z, sharding)

    return _build()


def time_fn(fn, *args, **kw):
    """Barrier-timed run.  Returns wall seconds."""
    # Barrier first so everybody starts together.
    try:
        multihost_utils.sync_global_devices("bench_start")
    except Exception:
        pass
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    jax.block_until_ready(out) if out is not None else None
    try:
        multihost_utils.sync_global_devices("bench_end")
    except Exception:
        pass
    return time.perf_counter() - t0


def gather_and_write(path: str, ds_name: str, A: jax.Array):
    """(G) variant — mirrors isdf_fitting.py's current zeta-write path."""
    gathered = multihost_utils.process_allgather(A, tiled=False)
    # process_allgather replicates across processes; with tiled=False we
    # get a leading process-dim that tilings across ranks.  Collapse it.
    if gathered.ndim == A.ndim + 1:
        # (world, n, n) → reassemble into (n, n) by slicing — for a
        # P('x','y') sharding on a (p,q) mesh, allgather with tiled=False
        # actually returns the full (n, n) on each process, replicated.
        gathered = np.asarray(jax.device_get(gathered))
        if gathered.shape[0] == A.shape[0]:
            pass  # shape already correct
    else:
        gathered = np.asarray(jax.device_get(gathered))

    if jax.process_index() == 0:
        import h5py
        with h5py.File(path, "a") as f:
            if ds_name in f:
                del f[ds_name]
            f.create_dataset(ds_name, data=gathered)
    # Sync so rank 0's write finishes before next iteration's allgather
    multihost_utils.sync_global_devices(f"gather_write_{ds_name}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8192,
                    help="Matrix side (complex128 → n*n*16 bytes).")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--path-ffi",
                    default="/pscratch/sd/j/jackm/bench_ffi.h5")
    ap.add_argument("--path-gather",
                    default="/pscratch/sd/j/jackm/bench_gather.h5")
    args = ap.parse_args()

    world = jax.process_count()
    if world == 16:
        p, q = 4, 4
    elif world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    _log(f"world={world}, grid=({p},{q}), n={args.n} c128, iters={args.iters}")
    bytes_total = args.n * args.n * 16
    _log(f"matrix size: {bytes_total / 1e9:.2f} GB global, "
         f"{bytes_total / world / 1e9:.3f} GB / rank")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # ---- FFI path (warmup + iters) ----
    _log("\n--- (F) FFI write_sharded_slab ---")
    fh = open_file(args.path_ffi, mesh=mesh, mode="w")
    # warmup
    W = build_matrix_c128(args.n, seed=-1, mesh=mesh)
    wtok = write_sharded_slab(fh, ds_name="warmup", A=W, mesh=mesh)
    jax.block_until_ready(wtok)
    _log("  warmup done")

    ffi_times = []
    for k in range(args.iters):
        A = build_matrix_c128(args.n, seed=k, mesh=mesh)
        jax.block_until_ready(A)   # isolate the write from the build
        dt = time_fn(lambda: write_sharded_slab(fh, ds_name=f"F{k}",
                                                 A=A, mesh=mesh))
        ffi_times.append(dt)
        _log(f"  iter {k}: {dt*1000:.1f} ms  "
             f"({bytes_total / dt / 1e9:.2f} GB/s global)")
    close_file(fh)

    # ---- Gather path (warmup + iters) ----
    _log("\n--- (G) process_allgather + rank-0 h5py write ---")
    if jax.process_index() == 0:
        import h5py
        with h5py.File(args.path_gather, "w") as f:
            pass  # truncate

    # warmup — forces the gather path JIT/alloc/etc
    Aw = build_matrix_c128(args.n, seed=-2, mesh=mesh)
    jax.block_until_ready(Aw)
    gather_and_write(args.path_gather, "warmup", Aw)
    _log("  warmup done")

    gather_times = []
    for k in range(args.iters):
        A = build_matrix_c128(args.n, seed=1000 + k, mesh=mesh)
        jax.block_until_ready(A)
        dt = time_fn(lambda: gather_and_write(
            args.path_gather, f"G{k}", A))
        gather_times.append(dt)
        _log(f"  iter {k}: {dt*1000:.1f} ms  "
             f"({bytes_total / dt / 1e9:.2f} GB/s global)")

    # ---- Summary ----
    _log("\n=== Summary ===")
    ffi_mean = float(np.mean(ffi_times))
    gth_mean = float(np.mean(gather_times))
    _log(f"  FFI   mean:    {ffi_mean*1000:7.1f} ms   "
         f"({bytes_total/ffi_mean/1e9:5.2f} GB/s)")
    _log(f"  Gather mean:   {gth_mean*1000:7.1f} ms   "
         f"({bytes_total/gth_mean/1e9:5.2f} GB/s)")
    if gth_mean > 0:
        _log(f"  FFI speedup:   {gth_mean/ffi_mean:.2f}x")

    if jax.process_index() == 0:
        for p_ in (args.path_ffi, args.path_gather):
            try: os.unlink(p_)
            except Exception: pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

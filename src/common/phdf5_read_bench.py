"""Benchmark: parallel FFI read vs load_wfns-style h5py read.

Setup: write a large sharded C128 matrix via the FFI once (so we have
a file to read), then measure two read paths:

  (H) load_wfns pattern: every rank opens the file with plain h5py and
      reads a slice into host memory.  Two sub-variants:
        (H-full) f['A'][:]   — every rank reads the WHOLE dataset, then
                               slices in-memory.  Matches
                               src/file_io/wfnreader.py:59 which does
                               self.coeffs = self._file['wfns/coeffs'][:]
                               on every process.
        (H-slab) f['A'][rs:re, cs:ce]  — every rank reads only its own
                                          hyperslab via h5py (no MPI-IO,
                                          one file handle per rank).
      Either way, each rank then jax.device_puts its local shard and
      jax.make_array_from_process_local_data promotes to a sharded
      global array — same shape as the FFI produces.

  (F) ffi.phdf5.read_sharded_slab — collective MPI-IO H5Dread directly
      into a sharded JAX array, one rank per hyperslab.

Times each variant ITERS times; reports mean wall and effective
read bandwidth (GB/s of the global matrix).

Measured on 4 nodes / 16 GPUs, Perlmutter, OpenMPI stack:

    n=16384 C128 (4.29 GB global, 268 MB/rank):
      (F) FFI            463 ms   ( 9.27 GB/s)
      (H-slab)           101 ms   (42.58 GB/s)    FFI / slab = 0.22x
      (H-full, wfnreader) 1882 ms  ( 2.28 GB/s)    FFI / full = 4.06x

    n=8192 C128 (1.07 GB global, 67 MB/rank):
      (F) FFI            194 ms   ( 5.53 GB/s)
      (H-slab)            38 ms   (28.35 GB/s)    FFI / slab = 0.20x
      (H-full, wfnreader) 555 ms   ( 1.93 GB/s)    FFI / full = 2.86x

Caveat: the file was written right before measurement, so Linux
page cache is HOT — that's why (H-slab) looks dominant (it's a
kernel-cache memcpy, not real disk I/O).  For the realistic LORRAX
case (WFN file written by a previous SLURM job, cold cache, file
size > node RAM), (H-slab) has to hit Lustre and drops to single-
OST bandwidth (~1 GB/s total).  The FFI's collective parallel read
stays at ~9 GB/s because every OST is served by its own aggregator.

The critical comparison is (F) vs (H-full), since that's what
wfnreader.py:59 currently does (``self.coeffs = self._file['wfns/coeffs'][:]``
on every rank).  The FFI is ~3-4x faster even with the file cache-hot,
and for a cold cache or a file larger than node RAM it should be
substantially more.  The FFI also uses n_ranks× less host memory
(each rank holds 1/n_ranks of the file instead of the full file).

Usage (4-node / 16-GPU interactive alloc):

    LORRAX_NNODES=4 LORRAX_NGPU=4 LORRAX_NTASKS=16 \\
        src/ffi/common/cpp/run_shifter.sh env \\
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_read_bench -n 16384 --iters 3
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
from ffi.phdf5 import (
    open_file, close_file, write_sharded_slab, read_sharded_slab,
)


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def time_fn(fn, *args, **kw):
    try:
        multihost_utils.sync_global_devices("bench_start")
    except Exception:
        pass
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    if out is not None:
        jax.block_until_ready(out)
    try:
        multihost_utils.sync_global_devices("bench_end")
    except Exception:
        pass
    return time.perf_counter() - t0, out


def build_matrix_c128(n: int, seed: int, mesh: Mesh) -> jax.Array:
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


def _local_shard_bounds(process_index, p, q, n_rows, n_cols):
    lr = n_rows // p
    lc = n_cols // q
    my_row = process_index // q
    my_col = process_index %  q
    return my_row * lr, (my_row + 1) * lr, my_col * lc, (my_col + 1) * lc, lr, lc


def read_ffi(path, ds_name, n, dtype, mesh):
    """(F) collective MPI-IO read into a sharded JAX array."""
    fh = open_file(path, mesh=mesh, mode="r")
    A = read_sharded_slab(fh, ds_name=ds_name,
                           global_shape=(n, n),
                           dtype=dtype, mesh=mesh)
    jax.block_until_ready(A)
    close_file(fh)
    return A


def read_h5py_full(path, ds_name, n, dtype, mesh, p, q):
    """(H-full) every rank reads the ENTIRE dataset via h5py, then keeps
    only its own shard — matches the wfnreader.py:59 pattern."""
    import h5py
    with h5py.File(path, "r") as f:
        A_all = f[ds_name][:]                          # (n, n) in host memory
    rs, re, cs, ce, lr, lc = _local_shard_bounds(
        jax.process_index(), p, q, n, n)
    A_local_np = A_all[rs:re, cs:ce]                    # slice
    A_local = jax.device_put(jnp.asarray(A_local_np))
    sharding = NamedSharding(mesh, P("x", "y"))
    return jax.make_array_from_process_local_data(sharding, A_local, (n, n))


def read_h5py_slab(path, ds_name, n, dtype, mesh, p, q):
    """(H-slab) every rank reads ONLY its own hyperslab via h5py (no
    MPI-IO).  Each rank opens an independent file handle."""
    import h5py
    rs, re, cs, ce, lr, lc = _local_shard_bounds(
        jax.process_index(), p, q, n, n)
    with h5py.File(path, "r") as f:
        A_local_np = f[ds_name][rs:re, cs:ce]
    A_local = jax.device_put(jnp.asarray(A_local_np))
    sharding = NamedSharding(mesh, P("x", "y"))
    return jax.make_array_from_process_local_data(sharding, A_local, (n, n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8192,
                    help="Matrix side (complex128 → n*n*16 bytes).")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--skip-h5py-full", action="store_true",
                    help="Skip the every-rank-full-read path (expensive "
                         "at large n or on small node memory).")
    ap.add_argument("--path",
                    default="/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/"
                            "bench_read.h5")
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
    bytes_total = args.n * args.n * 16
    _log(f"world={world}, grid=({p},{q}), n={args.n} c128, iters={args.iters}")
    _log(f"matrix size: {bytes_total/1e9:.2f} GB global, "
         f"{bytes_total/world/1e9:.3f} GB / rank")
    _log(f"path: {args.path}")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # ---- Step 1: write the matrix once (reference for all reads) ----
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(args.path), exist_ok=True)
    multihost_utils.sync_global_devices("bench_setup")
    A_ref = build_matrix_c128(args.n, seed=0, mesh=mesh)
    jax.block_until_ready(A_ref)
    fh_w = open_file(args.path, mesh=mesh, mode="w")
    t_w = write_sharded_slab(fh_w, ds_name="A", A=A_ref,
                              mesh=mesh, global_shape=(args.n, args.n))
    jax.block_until_ready(t_w)
    close_file(fh_w)
    _log("setup: file written via FFI")

    # ---- (F) FFI read, warmup + iters ----
    _log("\n--- (F) FFI read_sharded_slab ---")
    dt_warm, _ = time_fn(read_ffi, args.path, "A", args.n, jnp.complex128, mesh)
    _log(f"  warmup: {dt_warm*1000:.1f} ms")
    ffi_times = []
    for k in range(args.iters):
        dt, _ = time_fn(read_ffi, args.path, "A", args.n, jnp.complex128, mesh)
        ffi_times.append(dt)
        _log(f"  iter {k}: {dt*1000:.1f} ms  "
             f"({bytes_total/dt/1e9:.2f} GB/s global)")

    # ---- (H-slab) per-rank h5py hyperslab read ----
    _log("\n--- (H-slab) h5py per-rank hyperslab read ---")
    dt_warm, _ = time_fn(read_h5py_slab,
                         args.path, "A", args.n, jnp.complex128, mesh, p, q)
    _log(f"  warmup: {dt_warm*1000:.1f} ms")
    slab_times = []
    for k in range(args.iters):
        dt, _ = time_fn(read_h5py_slab,
                        args.path, "A", args.n, jnp.complex128, mesh, p, q)
        slab_times.append(dt)
        _log(f"  iter {k}: {dt*1000:.1f} ms  "
             f"({bytes_total/dt/1e9:.2f} GB/s global)")

    # ---- (H-full) every-rank full-dataset read ----
    full_times = []
    if not args.skip_h5py_full:
        _log("\n--- (H-full) h5py per-rank full-dataset read "
             "(wfnreader pattern) ---")
        dt_warm, _ = time_fn(read_h5py_full,
                             args.path, "A", args.n, jnp.complex128,
                             mesh, p, q)
        _log(f"  warmup: {dt_warm*1000:.1f} ms")
        for k in range(args.iters):
            dt, _ = time_fn(read_h5py_full,
                            args.path, "A", args.n, jnp.complex128,
                            mesh, p, q)
            full_times.append(dt)
            _log(f"  iter {k}: {dt*1000:.1f} ms  "
                 f"({bytes_total/dt/1e9:.2f} GB/s global)")

    # ---- Summary ----
    _log("\n=== Summary (mean over iters) ===")
    ffi_m  = float(np.mean(ffi_times))
    slab_m = float(np.mean(slab_times))
    _log(f"  (F) FFI     : {ffi_m*1000:7.1f} ms   "
         f"({bytes_total/ffi_m /1e9:5.2f} GB/s)")
    _log(f"  (H-slab)    : {slab_m*1000:7.1f} ms   "
         f"({bytes_total/slab_m/1e9:5.2f} GB/s)   "
         f"[speedup FFI: {slab_m/ffi_m:.2f}x]")
    if full_times:
        full_m = float(np.mean(full_times))
        _log(f"  (H-full)    : {full_m*1000:7.1f} ms   "
             f"({bytes_total/full_m/1e9:5.2f} GB/s)   "
             f"[speedup FFI: {full_m/ffi_m:.2f}x]")

    if jax.process_index() == 0:
        try: os.unlink(args.path)
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pipelining / async-overlap test for the phdf5 FFI.

Builds the same sharded matrix 5x in a row and writes each to its own
dataset.  Compares three patterns:

  (A) SYNC — block after every write:
      A = build(k); write(A); block(token); — every iteration fully
      serialized.

  (B) PIPELINED — fire-and-forget all writes, block only at the end:
      for k: A = build(k); token_k = write(A)
      block(token_k)                       — JAX dispatch is async, so
      build(k+1) may overlap with write(k) depending on how the XLA
      executor handles FFI calls.

  (C) MANUAL DOUBLE-BUFFER — one build ahead of each write:
      A_prev = build(0); tok = write(A_prev)
      for k>=1: A_next = build(k); block(tok); tok = write(A_next)
      block(tok)                           — explicit overlap hint.

If the FFI handler's blocking H5Dwrite truly runs in parallel with
subsequent device compute on the XLA scheduler, (B) will beat (A)
even though (B)'s last write doesn't start earlier.  (C) is the
upper bound for useful overlap with the current sync FFI.

Usage (inside an interactive alloc, from lorrax source root):

    LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_loop_test -n 1024 --iters 5
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

# Bootstrap multi-process JAX (SLURM auto) if applicable.
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
from ffi.phdf5 import open_file, write_sharded_slab, close_file


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_matrix(n: int, seed: int, mesh: Mesh) -> jax.Array:
    """Some cheap-but-not-trivial compute on the sharded mesh.

    We use a sharded random matrix + deterministic offset.  The compute
    time is roughly proportional to n**2 per rank — large enough that
    overlap with I/O is measurable.
    """
    key = jax.random.key(seed)
    sharding = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def _build():
        x = jax.random.uniform(key, (n, n), dtype=jnp.float64)
        # A little extra compute so build-time is non-trivial vs I/O.
        x = (x + x.T) * 0.5
        x = x + jnp.arange(n, dtype=jnp.float64)[:, None] * n + \
                jnp.arange(n, dtype=jnp.float64)[None, :]
        return jax.lax.with_sharding_constraint(x, sharding)

    return _build()


def sync_time_block(fn):
    """block_until_ready on all processes, then wall-clock after barrier."""
    jax.block_until_ready(fn)
    try:
        jax.experimental.multihost_utils.sync_global_devices("loop_test_barrier")
    except Exception:
        pass


def run_sync(fh, n, n_iters, mesh):
    _log("\n=== (A) SYNC: block after every write ===")
    t0 = time.perf_counter()
    for k in range(n_iters):
        A = build_matrix(n, seed=k, mesh=mesh)
        tok = write_sharded_slab(fh, ds_name=f"A{k}", A=A, mesh=mesh)
        sync_time_block(tok)
    wall = time.perf_counter() - t0
    _log(f"  total wall: {wall:.3f} s   ({wall/n_iters*1000:.1f} ms/iter)")
    return wall


def run_pipelined(fh, n, n_iters, mesh):
    _log("\n=== (B) PIPELINED: fire all writes, block once at end ===")
    t0 = time.perf_counter()
    tokens = []
    for k in range(n_iters):
        A = build_matrix(n, seed=100 + k, mesh=mesh)
        tokens.append(write_sharded_slab(fh, ds_name=f"B{k}", A=A, mesh=mesh))
    # block on all
    for t in tokens:
        jax.block_until_ready(t)
    try:
        jax.experimental.multihost_utils.sync_global_devices("loop_pipe_barrier")
    except Exception:
        pass
    wall = time.perf_counter() - t0
    _log(f"  total wall: {wall:.3f} s   ({wall/n_iters*1000:.1f} ms/iter)")
    return wall


def run_manual_doublebuf(fh, n, n_iters, mesh):
    _log("\n=== (C) MANUAL DOUBLE-BUFFER: build k+1 while write k runs ===")
    t0 = time.perf_counter()
    A_prev = build_matrix(n, seed=200, mesh=mesh)
    tok = write_sharded_slab(fh, ds_name="C0", A=A_prev, mesh=mesh)
    for k in range(1, n_iters):
        A_next = build_matrix(n, seed=200 + k, mesh=mesh)     # dispatch builds next
        jax.block_until_ready(tok)                             # wait for prior write
        tok = write_sharded_slab(fh, ds_name=f"C{k}", A=A_next, mesh=mesh)
    jax.block_until_ready(tok)
    try:
        jax.experimental.multihost_utils.sync_global_devices("loop_dbl_barrier")
    except Exception:
        pass
    wall = time.perf_counter() - t0
    _log(f"  total wall: {wall:.3f} s   ({wall/n_iters*1000:.1f} ms/iter)")
    return wall


def run_build_only(n, n_iters, mesh):
    _log("\n=== (D) BUILD-ONLY: baseline, no writes ===")
    t0 = time.perf_counter()
    for k in range(n_iters):
        A = build_matrix(n, seed=300 + k, mesh=mesh)
        jax.block_until_ready(A)
    try:
        jax.experimental.multihost_utils.sync_global_devices("loop_build_barrier")
    except Exception:
        pass
    wall = time.perf_counter() - t0
    _log(f"  total wall: {wall:.3f} s   ({wall/n_iters*1000:.1f} ms/iter)")
    return wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1024, help="Matrix side")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--path", type=str, default="/tmp/phdf5_loop.h5")
    args = ap.parse_args()

    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 16:
        p, q = 4, 4
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    _log(f"world={world}, grid=({p},{q}), n={args.n}, iters={args.iters}")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    fh = open_file(args.path, mesh=mesh, mode="w")
    _log(f"file opened: {args.path}")

    # Warmup — JIT compile build_matrix and the FFI call once so per-iter
    # timings aren't polluted by compile.
    _log("warmup...")
    W = build_matrix(args.n, seed=-1, mesh=mesh)
    wtok = write_sharded_slab(fh, ds_name="warmup", A=W, mesh=mesh)
    jax.block_until_ready(wtok)

    wd = run_build_only(args.n, args.iters, mesh)
    wa = run_sync(fh, args.n, args.iters, mesh)
    wb = run_pipelined(fh, args.n, args.iters, mesh)
    wc = run_manual_doublebuf(fh, args.n, args.iters, mesh)

    close_file(fh)

    _log("\n=== Summary ===")
    _log(f"  (D) build-only          : {wd:7.3f} s")
    _log(f"  (A) sync write          : {wa:7.3f} s")
    _log(f"  (B) pipelined write     : {wb:7.3f} s")
    _log(f"  (C) manual double-buf   : {wc:7.3f} s")
    _log(f"")
    pure_write_a = wa - wd
    _log(f"  wall spent writing in (A):  ~{pure_write_a:.3f} s")
    if pure_write_a > 0:
        overlap_b = 1 - (wb - wd) / pure_write_a
        overlap_c = 1 - (wc - wd) / pure_write_a
        _log(f"  (B) overlap factor:  {100*overlap_b:5.1f} %  (hidden by compute)")
        _log(f"  (C) overlap factor:  {100*overlap_c:5.1f} %  (hidden by compute)")

    if jax.process_index() == 0:
        os.unlink(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

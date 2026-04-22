"""4-GPU round-trip test for the parallel-HDF5 FFI.

Each of the 4 JAX processes holds its P('x','y') shard of a 128x128 F64
matrix.  We write via ffi.phdf5 collectively, then:

  (1) rank 0 serially reads the file back with plain h5py and checks
      exact equality — validates the write path,
  (2) all 4 ranks collectively re-read the file via
      ffi.phdf5.read_sharded_slab into a fresh sharded array, and each
      rank checks its own shard matches the reference — validates the
      read path.

Usage:

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_write_test
"""
from __future__ import annotations

from runtime import set_default_env
set_default_env()

import argparse
import os
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from runtime import init_jax_distributed
init_jax_distributed()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ffi.phdf5 import (
    open_file, close_file, write_sharded_slab, read_sharded_slab,
)


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def make_det_matrix(n: int, dtype=jnp.float64) -> np.ndarray:
    """Deterministic (n,n) array with distinct entries for exact-match check."""
    row = np.arange(n, dtype=np.float64)
    A = row[:, None] * n + row[None, :]    # A[i,j] = i*n + j
    return A.astype(np.dtype(dtype))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--path", type=str,
                    default=None,
                    help="HDF5 file path (default: tempfile on rank 0)")
    args = ap.parse_args()

    n = args.n
    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    _log(f"world={world}, grid=({p},{q}), n={n}")

    # All ranks must agree on the path.  Rank 0 picks a tempfile name
    # and broadcasts it via the JAX KV store (same pattern as our NCCL
    # uid broadcast for cuSOLVERMp).
    if args.path:
        path = args.path
    else:
        from ffi.common.broadcast import broadcast_bytes
        if jax.process_index() == 0:
            fd, path = tempfile.mkstemp(suffix=".h5", prefix="lorrax_phdf5_")
            os.close(fd); os.unlink(path)
        else:
            path = ""
        # Fixed-size buffer of 256 bytes for the path.
        buf = np.zeros(256, dtype=np.uint8)
        if jax.process_index() == 0:
            enc = path.encode()
            buf[:len(enc)] = np.frombuffer(enc, dtype=np.uint8)
        buf = broadcast_bytes(buf, key="lorrax_phdf5_test/path/v0")
        path = bytes(buf).rstrip(b"\x00").decode()
    _log(f"file path: {path}")

    # ---- build the sharded array ----
    A_host = make_det_matrix(n, dtype=jnp.float64)
    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    A = jax.device_put(jnp.asarray(A_host), sharding)
    _log(f"sharded A local shape = "
         f"{[s.data.shape for s in A.addressable_shards][:1]}")

    # ---- open + write + close ----
    fh = open_file(path, mesh=mesh, mode="w")
    _log(f"phdf5 file opened, ctx handle = 0x{fh:x}")

    token = write_sharded_slab(fh, ds_name="A", A=A,
                                mesh=mesh, global_shape=(n, n))
    jax.block_until_ready(token)
    _log("write returned (host thread unblocked post-H5Dwrite)")

    close_file(fh)
    _log("file closed")

    # ---- (1) serial readback + check (rank 0 only) ----
    serial_pass = 1
    if jax.process_index() == 0:
        import h5py
        with h5py.File(path, "r") as f:
            A_read = f["A"][...]
        err_serial = float(np.max(np.abs(A_read - A_host)))
        print(f"[serial ] max |A_read - A_ref| = {err_serial:.3e}")
        if err_serial > 0.0:
            print("[serial ] FAIL: non-zero error on serial round-trip")
            serial_pass = 0
        else:
            print("[serial ] PASS: exact round-trip via h5py")

    # ---- (2) parallel collective read via the read FFI (all ranks) ----
    fh2 = open_file(path, mesh=mesh, mode="r")
    _log(f"phdf5 re-opened for reading, ctx handle = 0x{fh2:x}")

    A_back = read_sharded_slab(fh2, ds_name="A",
                                global_shape=(n, n),
                                dtype=jnp.float64, mesh=mesh)
    jax.block_until_ready(A_back)
    _log("read returned (host thread unblocked post-H5Dread)")

    # Each rank checks its OWN local shard against the reference.
    local_shards = A_back.addressable_shards
    local_errs = []
    for sh in local_shards:
        local_np = np.asarray(sh.data)
        # index in reference is determined by shard index
        row_idx, col_idx = sh.index
        ref_block = A_host[row_idx, col_idx]
        local_errs.append(float(np.max(np.abs(local_np - ref_block))))
    err_local = max(local_errs) if local_errs else 0.0

    # Reduce across processes so we get the global max error.  Use JAX's
    # multihost allgather — same pattern the phdf5_vs_gather bench uses.
    from jax.experimental import multihost_utils
    flag_local = jnp.asarray(np.int64(1 if err_local > 0.0 else 0))
    flags_all = np.asarray(
        multihost_utils.process_allgather(flag_local, tiled=False))
    fail_sum = int(np.sum(flags_all))

    close_file(fh2)
    if jax.process_index() == 0:
        print(f"[parallel] my-rank max |shard_read - shard_ref| = {err_local:.3e}")
        print(f"[parallel] # ranks with non-zero error = {fail_sum}")
        if fail_sum == 0 and serial_pass:
            print("PASS: exact round-trip (serial + parallel)")
            os.unlink(path)
            return 0
        else:
            print("FAIL: non-zero error in parallel read shards"
                  if fail_sum else
                  "FAIL: serial path failed")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

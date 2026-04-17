"""4-GPU round-trip test for the parallel-HDF5 FFI.

Each of the 4 JAX processes holds its P('x','y') shard of a 128x128 F64
matrix.  We write via ffi.phdf5 collectively, then rank 0 serially
reads the file back with plain h5py and checks exact equality.

Usage:

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_write_test
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import sys
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Multi-process bootstrap (SLURM-aware, LORRAX pattern).
_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                         os.environ.get("JAX_NUM_PROCESSES",
                         os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"

_maybe_init_jax_distributed()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ffi.phdf5 import open_file, write_sharded_slab, close_file


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

    # ---- serial readback + check (rank 0 only) ----
    if jax.process_index() == 0:
        import h5py
        with h5py.File(path, "r") as f:
            A_read = f["A"][...]
        err = float(np.max(np.abs(A_read - A_host)))
        print(f"max |A_read - A_ref| = {err:.3e}")
        if err > 0.0:
            print("FAIL: non-zero error on exact-value round-trip")
            return 1
        print("PASS: exact round-trip of deterministic matrix")
        os.unlink(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

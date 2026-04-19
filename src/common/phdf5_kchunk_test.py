"""Smoke test for the ``lorrax_phdf5_read_kchunk`` FFI handler.

Writes a synthetic (nb, ns, ng, 2) float64 dataset via the existing
phdf5 write FFI, then reads n_kchunk independently-located hyperslab
windows via the new kchunk handler (in order, out of order, duplicated),
and checks each shard bitwise-equals the reference slice.

Usage (4-GPU):

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        src/ffi/common/cpp/run_shifter.sh env \
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
        HDF5_USE_FILE_LOCKING=FALSE \
        python3 -u -m common.phdf5_kchunk_test
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import sys
import tempfile

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
        try: jax.distributed.initialize()
        except Exception: pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils

from ffi.phdf5 import open_file, close_file
from ffi.phdf5.read import read_kchunk_sharded
from file_io.slab_io import SlabIO


def _log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


def main() -> int:
    world = jax.process_count()
    assert world == 4, f"this test expects 4 processes; got {world}"
    p, q = 2, 2
    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # Synthetic WFN-like dataset: (nb, ns, ng, 2).  Values chosen so each
    # element is unique → bitwise check is meaningful.
    nb, ns, ng = 16, 2, 40,
    ng = 40
    A_host = np.arange(nb * ns * ng * 2, dtype=np.float64).reshape(nb, ns, ng, 2)

    # Path via broadcast from rank 0.
    from ffi.common.broadcast import broadcast_bytes
    if jax.process_index() == 0:
        fd, path = tempfile.mkstemp(suffix=".h5", prefix="phdf5_kchunk_")
        os.close(fd); os.unlink(path)
    else:
        path = ""
    buf = np.zeros(256, dtype=np.uint8)
    if jax.process_index() == 0:
        enc = path.encode()
        buf[:len(enc)] = np.frombuffer(enc, dtype=np.uint8)
    buf = broadcast_bytes(buf, key="phdf5_kchunk_test/path/v0")
    path = bytes(buf).rstrip(b"\x00").decode()
    _log(f"file path: {path}")

    # Write the synthetic dataset sharded on the band axis (combined x,y).
    # bpr = nb / (p*q) = 4 for our 16-band case.  Using P(('x','y'),...)
    # exercises the same sharding encoding the read_kchunk caller will use.
    sharding_write = NamedSharding(mesh, P(("x", "y"), None, None, None))
    A_dev = jax.device_put(jnp.asarray(A_host), sharding_write)

    with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=True) as io:
        io.create_dataset("coeffs", shape=(nb, ns, ng, 2), dtype=np.float64)
        io.write_slab("coeffs", A_dev, global_shape=(nb, ns, ng, 2))
    _log("wrote synthetic dataset via write FFI")

    # Define a k-chunk with 4 windows: first 3 in order, last out of order
    # and overlapping with one of the earlier ones.  Windows are size
    # (nb=16, ns=2, window_g=8, 2), so per_rank_file_shape has band dim 4.
    window_g = 8
    k_offsets = np.array([
        [0, 0,  0, 0],     # [0:16, :, 0:8, :]
        [0, 0,  8, 0],     # [0:16, :, 8:16, :]
        [0, 0, 24, 0],     # [0:16, :, 24:32, :]  (out of order vs next row)
        [0, 0, 16, 0],     # [0:16, :, 16:24, :]  (out of order)
    ], dtype=np.int64)
    n_kchunk = k_offsets.shape[0]
    per_rank_file_shape = (nb // (p * q), ns, window_g, 2)   # (4, 2, 8, 2)
    _log(f"n_kchunk={n_kchunk}  per_rank_file_shape={per_rank_file_shape}")

    fh_r = open_file(path, mesh=mesh, mode="r")

    fused = read_kchunk_sharded(
        fh_r, "coeffs",
        n_kchunk=n_kchunk,
        file_global_shape=(nb, ns, ng, 2),
        per_rank_file_shape=per_rank_file_shape,
        dtype=np.float64,
        mesh=mesh,
        file_partition_spec=P(("x", "y"), None, None, None),
    )

    offset_arr = jnp.asarray(k_offsets, dtype=jnp.int64)
    out = fused(offset_arr)
    jax.block_until_ready(out)
    _log(f"read OK: out.shape={out.shape}  sharding={out.sharding.spec}")

    # Bitwise check per-rank: each shard covers bands [r*bpr, (r+1)*bpr)
    # for every k in the chunk.  Reference slab:
    #   A_host[r*bpr:(r+1)*bpr, :, off_k:off_k+window_g, :]
    bpr = nb // (p * q)
    rank = jax.process_index()
    local_err = 0
    for sh in out.addressable_shards:
        local_np = np.asarray(sh.data)                 # (n_kchunk, bpr, ns, window_g, 2)
        _, row_idx = sh.index[0], sh.index[1]
        # sh.index has 5 entries; the 2nd is the band slice
        row_slice = sh.index[1]
        row_start = row_slice.start if row_slice.start else 0
        for k_idx in range(n_kchunk):
            off_k = int(k_offsets[k_idx, 2])
            ref = A_host[row_start:row_start + bpr, :, off_k:off_k + window_g, :]
            if not np.array_equal(local_np[k_idx], ref):
                err = float(np.max(np.abs(local_np[k_idx] - ref)))
                print(f"[rank {rank}] k_idx={k_idx} band_start={row_start} "
                      f"off_g={off_k} FAIL err={err}", flush=True)
                local_err = 1

    close_file(fh_r)

    flag_local = jnp.asarray(np.int64(local_err))
    flags_all = np.asarray(
        multihost_utils.process_allgather(flag_local, tiled=False))
    n_fail = int(np.sum(flags_all))

    if jax.process_index() == 0:
        if n_fail == 0:
            print("PASS: kchunk FFI bit-identical on 4 ranks × 4 offsets "
                  "(in-order + out-of-order + overlapping)")
            try: os.unlink(path)
            except Exception: pass
            return 0
        print(f"FAIL: {n_fail} rank(s) mismatched")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

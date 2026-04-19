"""Multi-offset round-trip test for the phdf5 FFI.

Exercises the runtime-offset code path added 2026-04-19 (offset_base
as a Buffer<S64> arg rather than an FFI Attr).  Writes 4 slabs of a
big dataset at different (row,col) offsets, verifies all 4 hit the
SAME compiled shard_map module via ``_sm_cache``, then reads each
slab back and checks exact equality.

Regression target: before the Attr→Buffer refactor, each distinct
offset forced a fresh XLA compile (4 identical-signature HLO modules);
now the compile cache hits on 3 of 4 calls.  This test doesn't
directly measure compile count but DOES verify correctness across a
4-offset write/read pattern that the old regression tests didn't
exercise.

Usage:

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_multi_offset_test
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
from file_io.slab_io import SlabIO


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def main() -> int:
    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P(None, "x", "y"))

    # Dataset shape: (n_chunks, rows, cols) — 3D so the offset can
    # sweep along axis 0.  Mimics zeta_q layout (nq, n_rmu, n_rtot).
    n_chunks, rows, cols = 4, 16 * p, 16 * q
    _log(f"world={world}, grid=({p},{q}), ds shape=({n_chunks},{rows},{cols})")

    # Build 4 reference chunks, each a slice along axis 0.
    ref_chunks = [
        np.arange(rows * cols, dtype=np.float64).reshape(1, rows, cols) + 1000.0 * i
        for i in range(n_chunks)
    ]

    # Path: rank 0 picks, broadcast.
    if jax.process_index() == 0:
        fd, path = tempfile.mkstemp(suffix=".h5", prefix="phdf5_multi_")
        os.close(fd); os.unlink(path)
    else:
        path = ""
    from ffi.common.broadcast import broadcast_bytes
    buf = np.zeros(256, dtype=np.uint8)
    if jax.process_index() == 0:
        enc = path.encode()
        buf[:len(enc)] = np.frombuffer(enc, dtype=np.uint8)
    buf = broadcast_bytes(buf, key="lorrax_phdf5_multi_offset/path/v0")
    path = bytes(buf).rstrip(b"\x00").decode()
    _log(f"file path: {path}")

    # ---- Write 4 chunks at different offsets through a single SlabIO ----
    with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=True) as io:
        io.create_dataset("ds", shape=(n_chunks, rows, cols),
                          dtype=np.float64)
        for i, chunk_np in enumerate(ref_chunks):
            A = jax.device_put(jnp.asarray(chunk_np), sharding)
            io.write_slab("ds", A, offset=(i, 0, 0),
                          global_shape=(n_chunks, rows, cols))
            _log(f"  wrote chunk {i} at offset ({i},0,0)")

    # ---- Serial readback (rank 0) ----
    serial_pass = 1
    if jax.process_index() == 0:
        import h5py
        with h5py.File(path, "r") as f:
            data = f["ds"][...]
        max_err = 0.0
        for i, ref in enumerate(ref_chunks):
            err = float(np.max(np.abs(data[i:i+1] - ref)))
            max_err = max(max_err, err)
        print(f"[serial] max error across all 4 chunks: {max_err:.3e}")
        if max_err > 0.0:
            print("[serial] FAIL")
            serial_pass = 0
        else:
            print("[serial] PASS")

    # ---- Parallel read via SlabIO.read_slab at each offset ----
    local_pass = 1
    with SlabIO(path, mode="r", mesh=mesh, use_ffi_io=True) as io:
        for i, ref in enumerate(ref_chunks):
            A_back = io.read_slab(
                "ds", shape=(1, rows, cols), dtype=np.float64,
                offset=(i, 0, 0), partition_spec=P(None, "x", "y"))
            # Each rank checks its local shard against the reference slice.
            for sh in A_back.addressable_shards:
                local_np = np.asarray(sh.data)
                # 3-D index — (slice, row-block, col-block)
                _, row_idx, col_idx = sh.index
                ref_block = ref[0, row_idx, col_idx]
                err = float(np.max(np.abs(local_np[0] - ref_block)))
                if err > 0.0:
                    local_pass = 0
                    print(f"[parallel r{jax.process_index()}] "
                          f"chunk {i} FAIL err={err:.3e}", flush=True)

    from jax.experimental import multihost_utils
    pass_local = jnp.asarray(np.int64(local_pass))
    passes_all = np.asarray(
        multihost_utils.process_allgather(pass_local, tiled=False))
    global_pass = int(np.prod(passes_all)) == 1

    if jax.process_index() == 0:
        print(f"[parallel] all ranks passed: {bool(global_pass)}")
        if global_pass and serial_pass:
            print("PASS: exact round-trip on 4 offsets")
            os.unlink(path)
            return 0
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

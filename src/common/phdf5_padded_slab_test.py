"""PHDF5 SlabIO padded-slab round-trip test.

Exercises ``SlabIO.write_slab/read_slab(..., valid_shape=...)`` where
the physical JAX array is evenly block-sharded but the logical file
extent is ragged on the last rank(s).

Usage:

    LORRAX_NGPU=4 src/ffi/common/cpp/run_shifter.sh env \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        python3 -u -m common.phdf5_padded_slab_test
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


def _maybe_init_jax_distributed() -> None:
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get(
            "JAX_NUM_PROCESSES",
            os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"


_maybe_init_jax_distributed()

from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.slab_io import SlabIO


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _broadcast_path() -> str:
    from ffi.common.broadcast import broadcast_bytes
    if jax.process_index() == 0:
        fd, path = tempfile.mkstemp(suffix=".h5", prefix="phdf5_padded_")
        os.close(fd)
        os.unlink(path)
    else:
        path = ""
    buf = np.zeros(256, dtype=np.uint8)
    if jax.process_index() == 0:
        enc = path.encode()
        buf[:len(enc)] = np.frombuffer(enc, dtype=np.uint8)
    buf = broadcast_bytes(buf, key="lorrax_phdf5_padded/path/v0")
    return bytes(buf).rstrip(b"\x00").decode()


def main() -> int:
    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1

    physical_shape = (3 * p, 3 * q)
    valid_shape = (physical_shape[0] - 1, physical_shape[1] - 1)
    _log(
        f"world={world}, grid=({p},{q}), physical={physical_shape}, "
        f"valid={valid_shape}")

    host = np.arange(
        physical_shape[0] * physical_shape[1], dtype=np.float64
    ).reshape(physical_shape)
    expected_file = host[:valid_shape[0], :valid_shape[1]]
    expected_padded = np.zeros(physical_shape, dtype=np.float64)
    expected_padded[:valid_shape[0], :valid_shape[1]] = expected_file

    mesh = Mesh(np.asarray(jax.devices()).reshape(p, q), ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    A = jax.device_put(jnp.asarray(host), sharding)
    path = _broadcast_path()
    _log(f"file path: {path}")

    with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=True) as io:
        io.write_slab(
            "A",
            A,
            global_shape=valid_shape,
            valid_shape=valid_shape,
        )

    serial_pass = 1
    if jax.process_index() == 0:
        import h5py
        with h5py.File(path, "r") as f:
            got = f["A"][...]
        err = float(np.max(np.abs(got - expected_file)))
        print(f"[serial] max |file - expected| = {err:.3e}", flush=True)
        serial_pass = int(err == 0.0)

    with SlabIO(path, mode="r", mesh=mesh, use_ffi_io=True) as io:
        A_back = io.read_slab(
            "A",
            shape=physical_shape,
            dtype=np.float64,
            valid_shape=valid_shape,
            partition_spec=P("x", "y"),
        )
    A_back.block_until_ready()

    local_pass = 1
    for sh in A_back.addressable_shards:
        got = np.asarray(sh.data)
        row_idx, col_idx = sh.index
        ref = expected_padded[row_idx, col_idx]
        if float(np.max(np.abs(got - ref))) != 0.0:
            local_pass = 0
            print(
                f"[parallel r{jax.process_index()}] mismatch at {sh.index}",
                flush=True)

    flag = jnp.asarray(np.int64(local_pass))
    flags = np.asarray(multihost_utils.process_allgather(flag, tiled=False))
    parallel_pass = int(np.prod(flags)) == 1

    if jax.process_index() == 0:
        if serial_pass and parallel_pass:
            print("PASS: padded PHDF5 SlabIO round-trip", flush=True)
            os.unlink(path)
            return 0
        print(
            f"FAIL: serial_pass={bool(serial_pass)} "
            f"parallel_pass={parallel_pass}",
            flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
